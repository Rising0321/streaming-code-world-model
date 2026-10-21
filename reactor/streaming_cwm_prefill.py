"""Inference-only FA4 implementation of the native diffusion-forcing mask.

Uses the upstream DenseDF prefix decomposition, including the condition-only
case, and explicitly preserves the sampler's per-layer KV-store side effect.
This changes floating-point reduction order, not the set of attended tokens.
"""

from dataclasses import dataclass
from functools import lru_cache
from types import MethodType

import torch


@dataclass(frozen=True)
class PrefixMask:
    head_rows: int
    duv_rows: int
    audio_rows: int
    n_chunks: int
    static_sees_first: bool


def prefix_attention(q, k, v, layout, flash):
    from rtcwm.model.layout import NUM_CHUNKS
    from rtcwm.model.masks import CHUNK_ROWS

    h, a = layout.head_rows, layout.audio_rows
    dr = layout.duv_rows // NUM_CHUNKS
    d1, v0 = h + layout.duv_rows, h + layout.duv_rows + a
    if layout.n_chunks not in (0, 1) or q.shape[1] != v0 + layout.n_chunks * CHUNK_ROWS:
        raise ValueError("Prefix prefill only supports the native inference layouts")
    if layout.static_sees_first and not layout.n_chunks:
        raise ValueError("A first-span prefill must include video chunk zero")

    def static(t):
        return torch.cat([t[:, :h], t[:, d1:v0]], 1)

    kp, vp = (torch.cat([static(t), t[:, h:d1]], 1) for t in (k, v))
    st = h + a
    if layout.n_chunks:
        km, vm = (
            torch.cat([p[:, : st + dr], t[:, v0:]], 1) for p, t in ((kp, k), (vp, v))
        )
    else:
        km, vm = kp[:, :st], vp[:, :st]
    stop = st + (dr + CHUNK_ROWS if layout.static_sees_first else 0)
    out_static = flash(static(q), km[:, :stop], vm[:, :stop])
    out_proxy = [
        flash(
            q[:, h + p * dr : h + (p + 1) * dr],
            kp[:, : st + (p + 1) * dr],
            vp[:, : st + (p + 1) * dr],
        )
        for p in range(NUM_CHUNKS)
    ]
    out_video = [flash(q[:, v0:], km, vm)] if layout.n_chunks else []
    return torch.cat([out_static[:, :h], *out_proxy, out_static[:, h:], *out_video], 1)


def install_prefix_prefill(model):
    from rtcwm.model import attention, sampler
    from rtcwm.model.layout import NUM_CHUNKS

    original_mask = sampler.df_mask

    @lru_cache(maxsize=32)
    def layout(h, d, a, n, first):
        return PrefixMask(h, d, a, n, first)

    def mask(h, d, a, n, device, first_chunk=0, static_sees_first=True):
        if n in (0, 1) and not first_chunk and d % NUM_CHUNKS == 0:
            return layout(h, d, a, n, static_sees_first)
        return original_mask(h, d, a, n, device, first_chunk, static_sees_first)

    sampler.df_mask = mask
    for block in model.blocks:
        layer = block.attn
        original = layer.forward

        def forward(self, hidden, rotation_table=None, _original=original):
            mask = attention._state["mask"]
            if not isinstance(mask, PrefixMask):
                return _original(hidden, rotation_table)
            if torch.is_grad_enabled():
                raise RuntimeError("Prefix prefill is inference-only")
            b, s, _ = hidden.shape
            q, k, v = self.qkv_proj(hidden).split(self.inner_dim, -1)
            q = self.q_norm(q.reshape(b, s, self.num_heads, self.head_dim))
            k = self.k_norm(k.reshape(b, s, self.num_heads, self.head_dim))
            v = v.reshape(b, s, self.num_heads, self.head_dim)
            if rotation_table is not None:
                q, k = (
                    attention.rope(q, rotation_table),
                    attention.rope(k, rotation_table),
                )
            dtype = (
                hidden.dtype
                if hidden.dtype in (torch.bfloat16, torch.float16)
                else torch.bfloat16
            )
            q, k, v = (t.to(dtype) for t in (q, k, v))
            kv = attention._state["kv"]
            if kv is not None and kv.get("store") is not None:
                kv["store"].append(
                    (k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
                )
            output = prefix_attention(q, k, v, mask, attention._fa3)
            return self.out_proj(output.reshape(b, s, self.inner_dim))

        layer.forward = MethodType(forward, layer)
