"""Loading the frozen H3 transformer with a LoRA (trainable or merged), and sharding its blocks."""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from .. import paths
from . import attention

EXPECTED_LORA_MODULES = 200            # the CWM LoRA: qkv / out / fc1 / fc2 of the 50 blocks


def load_base():
    """The released Ref2VA H3 transformer on CPU, bf16, frozen."""
    from musubi_tuner.minimax_h3.model import load_h3_transformer
    tr = load_h3_transformer(paths.DIT, device="cpu", dtype=torch.bfloat16, attn_mode="flash3", split_attn=False,
                             disable_mmap=False)
    tr.requires_grad_(False)
    return tr


def attach_trainable_lora(tr, lora: Path):
    """Wrap the blocks' linears with a LoRA initialised from `lora`. Returns the (fp32-ready) network."""
    from musubi_tuner.networks import lora_minimax_h3
    sd = load_file(str(lora))
    net = lora_minimax_h3.create_arch_network_from_weights(1.0, sd, unet=tr, for_inference=False)
    net.apply_to(None, tr, apply_text_encoder=False, apply_unet=True)
    info = net.load_state_dict(sd, strict=False)
    assert not info.unexpected_keys, info.unexpected_keys[:5]
    assert not info.missing_keys and len(net.unet_loras) == EXPECTED_LORA_MODULES, (len(info.missing_keys), len(net.unet_loras))
    return net


def _skip_when_off(m):
    full = m.forward

    def forward(x):
        return m.org_forward(x) if m.multiplier == 0 else full(x)
    return forward


class LoRAStack:
    """Several LoRAs on one frozen base (distillation: student, critic, teacher). They are stacked on
    the same linears; exactly one is active at a time (the others get multiplier 0). Activation
    checkpointing recomputes blocks in backward, so the network that built a graph must be active
    again when that graph is back-propagated."""

    def __init__(self, tr, specs: dict):
        """specs: name -> (initial LoRA file, trainable)."""
        self.nets, self.trainable = {}, {}
        for name, (path, trainable) in specs.items():
            net = attach_trainable_lora(tr, path)
            net.requires_grad_(trainable)
            self.nets[name], self.trainable[name] = net, trainable
            for m in net.unet_loras:                      # musubi's LoRA computes its branch even at
                m.forward = _skip_when_off(m)             # multiplier 0: every forward paid for all three
        self.active = None

    def to(self, device):
        for net in self.nets.values():
            net.to(device, torch.float32)
        return self

    def use(self, name: str):
        """Activate one LoRA; only it (if trainable) requires gradients, so a backward pass produces
        (and all-reduces) gradients for that network alone."""
        for n, net in self.nets.items():
            for m in net.unet_loras:
                m.multiplier = 1.0 if n == name else 0.0
            net.requires_grad_(n == name and self.trainable[n])
        self.active = name

    def params(self, name: str):
        return list(self.nets[name].parameters())


def load_for_inference(lora: Path, device, fp8: bool = False):
    """Base + merged LoRA on `device`, eval mode, causal attention installed; fp8: the blocks' large linears
    in FP8 (rtcwm.model.fp8)."""
    from cwm_h3_inference.runtime import strict_merge_lora
    tr = load_base()
    strict_merge_lora(tr, lora, EXPECTED_LORA_MODULES)
    tr = tr.to(device).eval()
    if fp8:
        from .fp8 import quantize_blocks
        quantize_blocks(tr)
    attention.install(tr)
    return tr


def shard_blocks(tr, group, device):
    """FSDP-shard each block over `group` (the GPUs of one node); everything else is small and replicated.
    The LoRA weights live in the network module, outside FSDP, and stay replicated."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    for name, child in tr.named_children():
        if name != "blocks":
            child.to(device)
    from torch.distributed.fsdp._runtime_utils import _reshard, _unshard

    # FSDP (v1) leaves a block's full parameters gathered after a forward run without autograd and after
    # a forward inside our activation checkpoint -- every block then stays gathered (+55-60 GB per GPU
    # during a KV-cache rollout, a no-grad teacher pass or a student gradient step). And as the base
    # weights are frozen, FSDP registers no post-backward hook either, so a block gathered for its
    # backward stays gathered as well. We therefore manage the block's gather/reshard ourselves:
    def in_backward():
        return torch._C._current_graph_task_id() != -1

    def gather_for_recompute(module, _inputs):
        # FSDP gathers a block for backward once per backward pass; a block used by several gradient
        # forwards (several student chunks) is recomputed again after we resharded it
        if in_backward() and module._handle.needs_unshard():
            _unshard(module, module._handle, module._unshard_stream, module._pre_unshard_stream)
            torch.cuda.current_stream().wait_stream(module._unshard_stream)

    def reshard_after_forward(module, inputs, _output):
        if in_backward():                  # checkpoint recomputation: the backward of the block follows
            return
        _reshard(module, module._handle, True)
        h = inputs[0] if inputs and isinstance(inputs[0], torch.Tensor) else None
        if h is not None and h.requires_grad:             # the block's backward is done once its input has a gradient
            h.register_hook(lambda g: (_reshard(module, module._handle, True), g)[1])

    # Every block is its own FSDP root, so FSDP's forward_prefetch (which follows one root's execution
    # order) never gathers the next block. Outside backward we do it by hand: once block i holds its
    # parameters (a pre-hook on the wrapped module, i.e. inside FSDP's forward, after its unshard), block
    # i+1's all-gather is issued on its own unshard stream and overlaps block i's compute; FSDP's
    # pre-forward of block i+1 then finds it prefetched and only waits for that stream. (The same in
    # backward -- block i-1 during block i's recomputation -- measured no faster: the backward is compute
    # bound.)
    from torch.distributed.fsdp._flat_param import HandleTrainingState

    def prefetch_next(i):
        def hook(_module, _inputs):
            if in_backward() or i + 1 >= len(tr.blocks):
                return
            nxt = tr.blocks[i + 1]
            h = nxt._handle
            if (h is None or getattr(nxt, "_unshard_stream", None) is None       # not run yet: FSDP initialises
                    or not hasattr(h.flat_param, "_full_param_padded")            # a block lazily, in its first
                    or h._prefetched or not h.needs_unshard()):                  # forward
                return
            prev, h._training_state = h._training_state, HandleTrainingState.FORWARD
            _unshard(nxt, h, nxt._unshard_stream, nxt._pre_unshard_stream)
            h._training_state, h._prefetched = prev, True
        return hook

    for i in range(len(tr.blocks)):
        tr.blocks[i] = FSDP(tr.blocks[i], process_group=group, device_id=device, use_orig_params=True,
                            sharding_strategy=ShardingStrategy.FULL_SHARD, limit_all_gathers=True)
        tr.blocks[i].register_forward_pre_hook(gather_for_recompute)
        tr.blocks[i].register_forward_hook(reshard_after_forward)
        tr.blocks[i].module.register_forward_pre_hook(prefetch_next(i))
    attention.install(tr)
    return tr
