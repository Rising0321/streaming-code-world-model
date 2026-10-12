"""FP8 linears for sampling (Hopper): the large bias-free linears of every DiT block -- attention qkv / out
and the MLP -- with e4m3 weights scaled per output channel and activations quantized per row on the fly,
multiplied by torch._scaled_mm into bf16. Inference only (no gradient), on a model whose LoRA is merged;
approximate (FP8 rounding), so its effect is measured, not assumed (tests/test_fp8.py).
"""
from __future__ import annotations

import torch
from torch import nn

E4M3_MAX = 448.0


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None
        w = linear.weight.detach().float()                                     # [out, in]
        scale = (w.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp_min(1e-12)
        self.register_buffer("weight_fp8", (w / scale).to(torch.float8_e4m3fn), persistent=False)
        self.register_buffer("weight_scale", scale.t().contiguous(), persistent=False)    # [1, out]
        self.out_features = linear.out_features

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        scale = (x2.abs().amax(dim=1, keepdim=True).float() / E4M3_MAX).clamp_min(1e-12)   # [rows, 1]
        xq = (x2.float() / scale).to(torch.float8_e4m3fn)
        y = torch._scaled_mm(xq, self.weight_fp8.t(), scale_a=scale, scale_b=self.weight_scale,
                             out_dtype=torch.bfloat16)
        return y.reshape(*shape[:-1], self.out_features)


def quantize_blocks(model) -> int:
    """Swap the DiT blocks' qkv / out / fc1 / fc2 linears for FP8Linear in place; returns how many."""
    n = 0
    for block in model.blocks:
        for parent, names in ((block.attn, ("qkv_proj", "out_proj")), (block.mlp, ("fc1", "fc2"))):
            for name in names:
                setattr(parent, name, FP8Linear(getattr(parent, name)))
                n += 1
    torch.cuda.empty_cache()
    return n
