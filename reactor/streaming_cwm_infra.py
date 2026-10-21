"""Inference acceleration preserving the released schedule, precision and history.

FA4 changes floating-point reduction order; it is not a bit-exact replacement.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist
from streaming_cwm_sampling import StreamingDecoder
from torch import nn


def install_fa4():
    """Optional dense-attention backend: mathematically equivalent, NOT bit-exact."""
    from flash_attn.cute import flash_attn_func
    from rtcwm.model import attention

    def flash(q, k, v):
        output = flash_attn_func(q, k, v, num_splits=1)
        return output[0] if isinstance(output, tuple) else output

    attention._fa3 = flash
    attention.KV_ATTENTION = "fa4"


def install_vae_fa4(vae):
    """Same decoder attention and FP32 Q/K normalization, with FA4 reductions."""
    from flash_attn.cute import flash_attn_func
    from musubi_tuner.minimax_h3.video_vae import Attention, _apply_rotary_emb

    def forward(self, hidden, rotary_emb=None):
        batch, length, _ = hidden.shape
        qkv = self.to_qkv(hidden).view(batch, length, self.heads, 3 * self.dim_head)
        query, key, value = qkv.chunk(3, -1)
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
        output = flash_attn_func(query, key, value, num_splits=1)
        if isinstance(output, tuple):
            output = output[0]
        return self.to_out(output.reshape(batch, length, -1))

    for module in vae.decoder.modules():
        if isinstance(module, Attention):
            module.forward = MethodType(forward, module)


class GraphTileDecoder(nn.Module):
    """Capture one native batch-one tile without changing kernels or precision."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.graph = None
        self.signature = None

    @torch.inference_mode()
    def forward(self, value):
        signature = (value.shape, value.stride(), value.dtype, value.device)
        if self.graph is not None and signature != self.signature:
            raise ValueError("Decoder shape/device changed; rebuild the CUDA graph")
        if self.graph is None:
            self.signature = signature
            self.input = torch.empty_like(value)
            self.input.copy_(value)
            stream = torch.cuda.Stream(device=value.device)
            stream.wait_stream(torch.cuda.current_stream(value.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.decoder(self.input)
            torch.cuda.current_stream(value.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.decoder(self.input)
        self.input.copy_(value)
        self.graph.replay()
        # Each spatial tile must retain its own output until native stitching.
        return self.output.clone()


class GPUBytesDecoder(StreamingDecoder):
    """Execute the *same* unfused FP32 RGB conversion before the host copy."""

    def _decode(self, latent):
        vae = self.vae
        latent = latent.to(dtype=torch.float16)
        latent = latent * vae.latents_std.view(1, -1, 1, 1, 1).to(
            latent
        ) + vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        clip = vae._decode_clip(latent)
        # A distributed decoder returns its stitched clip only on rank zero.
        if clip is None:
            return None
        frames = clip[:, :, 3:20]
        if self.overlap is not None:
            frames = vae._blend(self.overlap, frames, 5, dim=-3)
        self.overlap = clip[:, :, 23:].clone()
        pixels = frames.float() * vae.pixel_std.to(frames) + vae.pixel_mean.to(frames)
        pixels = pixels.clamp(0.0, 1.0) * 2.0 - 1.0
        pixels = pixels[0].float()
        return (
            ((pixels.clamp(-1, 1) + 1) * 127.5)
            .round()
            .byte()
            .permute(1, 2, 3, 0)
            .contiguous()
            .cpu()
            .numpy()
        )


def install_spatial_decode(vae, rank: int, world_size: int):
    """Distribute independent tiles, keeping batch=1 and native stitching order.

    All ranks must call push/finish, even though only rank zero returns pixels.
    Does not touch VAE encoding or its tile/boundary geometry.
    """
    original = vae._decode_clip

    @torch.inference_mode()
    def decode(self, latents):
        if world_size == 1:
            return original(latents)
        if not self.use_tiling or latents.shape[0] != 1:
            raise ValueError("Parallel decode requires the native batch-one tiled VAE")
        hs, hl, ho = self._split_tiles(latents.shape[-2] * self.vae_ratio)
        ws, wl, wo = self._split_tiles(latents.shape[-1] * self.vae_ratio)
        coords = [(y, h, x, w) for y, h in zip(hs, hl) for x, w in zip(ws, wl)]
        shapes = {(h, w) for _, h, _, w in coords}
        if len(shapes) != 1:
            raise ValueError(
                "Parallel decode requires equal native spatial tile shapes"
            )
        per_rank = (len(coords) + world_size - 1) // world_size
        padded = coords + [coords[-1]] * (per_rank * world_size - len(coords))
        outputs = []
        for y, h, x, w in padded[rank * per_rank : (rank + 1) * per_rank]:
            tile = latents[
                ...,
                y // self.vae_ratio : (y + h) // self.vae_ratio,
                x // self.vae_ratio : (x + w) // self.vae_ratio,
            ]
            # Preserve each native GEMM/convolution's shape, strides and dtype.
            outputs.append(self.decoder(self.post_quant_conv(tile)))
        local = torch.cat(outputs, dim=0).contiguous()
        all_tiles = torch.empty(
            (world_size * per_rank, *local.shape[1:]),
            device=local.device,
            dtype=local.dtype,
        )
        dist.all_gather_into_tensor(all_tiles, local)
        if rank:
            return None
        tiles = [
            [all_tiles[i * len(ws) + j : i * len(ws) + j + 1] for j in range(len(ws))]
            for i in range(len(hs))
        ]
        return self._stitch_tiles(tiles, ho, wo)

    vae._decode_clip = MethodType(decode, vae)


class _TimeBank(nn.Module):
    def __init__(self, entries):
        super().__init__()
        self.keys = {}
        self.index = None
        for index, (times, embedding) in enumerate(entries):
            self.keys[self.key(times)] = index
            self.register_buffer(f"embedding_{index}", embedding.detach().clone())

    @staticmethod
    def key(times):
        if times.ndim != 1 or times.dtype != torch.float32:
            raise ValueError("AdaLN bank requires native one-dimensional FP32 times")
        return tuple(times.detach().cpu().tolist())

    def forward(self, times):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Fixed-schedule AdaLN bank is inference-only")
        key = self.key(times)
        if key not in self.keys:
            raise ValueError(f"Uncached native schedule: {key}; rebuild the AdaLN bank")
        self.index = self.keys[key]
        return getattr(self, f"embedding_{self.index}")


class _ProjectionBank(nn.Module):
    def __init__(self, cursor, values):
        super().__init__()
        # Cursor is owned once by model.time_embedder; don't register it again.
        object.__setattr__(self, "cursor", cursor)
        self.parts = len(values[0])
        for index, parts in enumerate(values):
            for part, tensor in enumerate(parts):
                self.register_buffer(f"value_{index}_{part}", tensor.detach().clone())

    def forward(self, embeddings):
        if self.training or torch.is_grad_enabled() or self.cursor.index is None:
            raise RuntimeError(
                "AdaLN projection requires a selected inference schedule"
            )
        return tuple(
            getattr(self, f"value_{self.cursor.index}_{part}")
            for part in range(self.parts)
        )


@torch.inference_mode()
def install_adaln_bank(model):
    """Partial evaluation of the released 4-step sampler, never batching time sets.

    Native rtcwm.run_blocks casts time embeddings to BF16 *before* AdalnProj's
    SiLU. Preserve that order, even though other H3 implementations differ.
    Unknown times fail loudly. Reload the model after any weight/schedule change.
    """
    from musubi_tuner.minimax_h3.model import AdalnProj
    from rtcwm.model.layout import CLEAN_T, TEXT_T_FIXED, VISUAL_T
    from rtcwm.model.sampler import schedule

    if model.training:
        raise ValueError("Load frozen inference weights before caching AdaLN")
    device = model.video_patch_proj.weight.device
    sigmas = schedule(4, device, 5.0).cpu().tolist()
    times = []
    for sigma in sigmas[:-1]:
        times.extend(([1.0 - sigma], [1.0 - sigma, TEXT_T_FIXED, VISUAL_T, CLEAN_T]))
    times.append([TEXT_T_FIXED, VISUAL_T, CLEAN_T])
    unique = {}
    for values in times:
        value = torch.unique(torch.tensor(values, dtype=torch.float32), sorted=True).to(
            device
        )
        unique[_TimeBank.key(value)] = value
    entries = [(values, model.time_embedder(values)) for values in unique.values()]
    cursor = _TimeBank(entries).eval()
    replaced_bytes = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, AdalnProj):
            values = [module(embedding.to(model.dtype)) for _, embedding in entries]
            replaced_bytes += sum(
                p.numel() * p.element_size() for p in module.parameters()
            )
            parent_name, _, leaf = name.rpartition(".")
            setattr(
                model.get_submodule(parent_name),
                leaf,
                _ProjectionBank(cursor, values).eval(),
            )
    model.time_embedder = cursor
    return {"time_sets": len(entries), "replaced_gib": replaced_bytes / 2**30}


class PackedKV:
    """Pack each layer's immutable history once for a chunk's four steps.

    Native keys/RoPE values are copied exactly; only the current K/V tail changes.
    The scope is one sampling chunk, so no new history policy or stale cache is
    introduced. Each GPU spends extra memory to avoid three repeat packs.
    """

    def __init__(self, model):
        from rtcwm.model import attention

        self.attention = attention
        self.entries = {}
        self.enabled = False
        for block in model.blocks:
            module = block.attn
            original = module.forward

            def forward(layer, hidden, rotation_table=None, _original=original):
                state = attention._state
                kv = state["kv"]
                if (
                    not self.enabled
                    or torch.is_grad_enabled()
                    or state["mask"] is not None
                    or kv is None
                    or kv.get("causal_blocks")
                    or not kv.get("past")
                ):
                    return _original(hidden, rotation_table)
                b, s, _ = hidden.shape
                q, k, v = layer.qkv_proj(hidden).split(layer.inner_dim, dim=-1)
                q = layer.q_norm(q.reshape(b, s, layer.num_heads, layer.head_dim))
                k = layer.k_norm(k.reshape(b, s, layer.num_heads, layer.head_dim))
                v = v.reshape(b, s, layer.num_heads, layer.head_dim)
                if rotation_table is not None:
                    q, k = (
                        attention.rope(q, rotation_table),
                        attention.rope(k, rotation_table),
                    )
                dtype = (
                    hidden.dtype
                    if hidden.dtype in (torch.float16, torch.bfloat16)
                    else torch.bfloat16
                )
                q, k, v = (t.transpose(1, 2).contiguous().to(dtype) for t in (q, k, v))
                if kv.get("store") is not None:
                    kv["store"].append((k, v))
                pieces = kv["past"][layer._rtcwm_layer]
                entry = self.entries.get(layer._rtcwm_layer)
                if entry is None or entry[0] is not pieces:
                    full_k = torch.cat([attention._key(p) for p in pieces] + [k], 2)
                    full_v = torch.cat([p[1] for p in pieces] + [v], 2)
                    self.entries[layer._rtcwm_layer] = (pieces, full_k, full_v, s)
                else:
                    _, full_k, full_v, tail = entry
                    if s != tail:
                        raise ValueError(
                            "Current chunk shape changed inside a packed-KV scope"
                        )
                    full_k[:, :, -s:].copy_(k)
                    full_v[:, :, -s:].copy_(v)
                if attention.KV_ATTENTION == "sdpa":
                    output = torch.nn.functional.scaled_dot_product_attention(
                        q, full_k, full_v
                    ).transpose(1, 2)
                else:
                    output = attention._fa3(
                        q.transpose(1, 2),
                        full_k.transpose(1, 2),
                        full_v.transpose(1, 2),
                    )
                return layer.out_proj(output.reshape(b, s, layer.inner_dim))

            module.forward = MethodType(forward, module)

    @contextmanager
    def chunk(self):
        self.entries.clear()
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False
            self.entries.clear()
