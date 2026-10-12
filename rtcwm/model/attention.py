"""Masked / cached attention for the MiniMax-H3 blocks.

musubi's H3 Attention runs dense FlashAttention over the whole packed sequence. install() patches its
forward with two extra modes, selected per block call through `active(...)`:

  mask : torch flex_attention with a BlockMask over the whole packed sequence (training, reference)
  dense: a masks.DenseDF instead of a BlockMask -- the same training mask as FlashAttention-3 prefix calls
  kv   : the call's rows attend to cached keys/values of earlier rows plus their own (KV-cache sampling,
         FlashAttention-3; see KV_ATTENTION);
         with store=list the call's own keys/values are appended per layer. A mask may be combined with
         kv to fill the cache (chunk 0 together with the condition rows).

The mode is set inside each block call (not around the loop) so that activation-checkpoint
recomputation in backward sees it too.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from musubi_tuner.minimax_h3 import model as h3model

from .layout import CHUNK_SIZES
from .masks import CHUNK_ROWS, DenseDF

_compiled_flex = torch.compile(flex_attention, dynamic=True)
# KV-cache attention over the concatenated past: "fa3" FlashAttention-3 (23% faster per sampled chunk);
# "sdpa" torch SDPA (the earlier behaviour, kept for tests)
KV_ATTENTION = "fa3"
_state = {"mask": None, "kv": None}
_orig_forward = h3model.Attention.forward
_orig_rope = h3model._apply_rope_split_half              # kept for tests/test_samplers.py


def _forward(self, hidden_states, rotation_table=None):
    mask, kv = _state["mask"], _state["kv"]
    if mask is None and kv is None:
        return _orig_forward(self, hidden_states, rotation_table)
    b, s, _ = hidden_states.shape
    q, k, v = self.qkv_proj(hidden_states).split(self.inner_dim, dim=-1)
    q = self.q_norm(q.reshape(b, s, self.num_heads, self.head_dim))
    k = self.k_norm(k.reshape(b, s, self.num_heads, self.head_dim))
    v = v.reshape(b, s, self.num_heads, self.head_dim)
    if rotation_table is not None:
        q = rope(q, rotation_table)
        k = rope(k, rotation_table)
    dt = hidden_states.dtype if hidden_states.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    if isinstance(mask, DenseDF):
        out = dense_df(q.to(dt), k.to(dt), v.to(dt), mask)
        return self.out_proj(out.reshape(b, s, self.inner_dim))
    q, k, v = (t.transpose(1, 2).contiguous().to(dt) for t in (q, k, v))     # [B,H,S,D]
    if kv is not None and kv.get("store") is not None:
        kv["store"].append((k, v))
    if mask is not None:
        out = _compiled_flex(q, k, v, block_mask=mask)
    else:
        pieces = kv["past"][self._rtcwm_layer] if kv.get("past") is not None else []
        if kv.get("causal_blocks"):                  # the call's rows are blocks of n rows, block p seeing the
            n = kv["causal_blocks"]                  # past and blocks <= p (the proxy rows of a span)
            pk = torch.cat([_key(p) for p in pieces], 2).transpose(1, 2)
            pv = torch.cat([p[1] for p in pieces], 2).transpose(1, 2)
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
            out = torch.cat([_fa3(qs[:, i:i + n], torch.cat([pk, ks[:, :i + n]], 1), torch.cat([pv, vs[:, :i + n]], 1))
                             for i in range(0, s, n)], 1)
            return self.out_proj(out.reshape(b, s, self.inner_dim))
        if pieces:                                   # concatenated per layer: one layer's copy at a time;
            k = torch.cat([_key(p) for p in pieces] + [k], 2)   # a piece may carry a pending time shift
            v = torch.cat([p[1] for p in pieces] + [v], 2)
        if KV_ATTENTION == "sdpa":
            out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        else:
            out = _fa3(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.out_proj(out.reshape(b, s, self.inner_dim))
    return self.out_proj(out.transpose(1, 2).reshape(b, s, self.inner_dim))


def _key(piece):
    """Cached keys [B,H,S,D] of a piece (k, v) or (k, v, rotation): the rotation (rtcwm.model.sampler._moved)
    moves them on the time axis -- RoPE rotations compose -- without storing a moved copy."""
    if len(piece) == 2:
        return piece[0]
    return rope(piece[0].transpose(1, 2), piece[2]).transpose(1, 2)


def _fa3(q, k, v):
    from flash_attn_interface import flash_attn_func
    out = flash_attn_func(q, k, v)
    return out[0] if isinstance(out, tuple) else out


def dense_df(q, k, v, lay: DenseDF):
    """Attention of a packed training span [head | DUV 0..5 | audio | video 0..n-1] under df_mask, as
    unmasked FlashAttention-3 calls over key prefixes (see masks.DenseDF). q, k, v [B,S,H,D]."""
    h, a = lay.head_rows, lay.audio_rows
    dr, cr = lay.duv_rows // len(CHUNK_SIZES), CHUNK_ROWS
    d1 = h + lay.duv_rows
    v0 = d1 + a
    assert q.shape[1] == v0 + lay.n_chunks * cr, (q.shape, lay)
    duv = lambda t, p: t[:, h + p * dr:h + (p + 1) * dr]
    vid = lambda t, c: t[:, v0 + c * cr:v0 + (c + 1) * cr]

    def main(t):
        parts = [t[:, :h], t[:, d1:v0]]
        for c in range(lay.n_chunks):
            parts += [duv(t, c), vid(t, c)]
        return torch.cat(parts, 1)

    km, vm = main(k), main(v)
    kp, vp = (torch.cat([t[:, :h], t[:, d1:v0], t[:, h:d1]], 1) for t in (k, v))
    st = h + a                                                   # static keys lead both orders
    n = st + (dr + cr if lay.static_sees_first else 0)
    o_st = _fa3(torch.cat([q[:, :h], q[:, d1:v0]], 1), km[:, :n], vm[:, :n])
    o_duv = [_fa3(duv(q, p), kp[:, :st + (p + 1) * dr], vp[:, :st + (p + 1) * dr]) for p in range(len(CHUNK_SIZES))]
    o_vid = [_fa3(vid(q, c), km[:, :st + (c + 1) * (dr + cr)], vm[:, :st + (c + 1) * (dr + cr)])
             for c in range(lay.n_chunks)]
    return torch.cat([o_st[:, :h], *o_duv, o_st[:, h:], *o_vid], 1)


def rope(hidden_states, rotation_table):
    """musubi's _apply_rope_split_half written element-wise. The original multiplies every (token, head,
    pair) by its 2x2 rotation with a batched matmul -- one tiny cuBLAS gemv per few thousand elements,
    ~40% of a sampling step. Same rotation table, accumulated in fp32 like the matmul."""
    pairs = rotation_table.shape[-3]
    r = rotation_table.float()                                  # [B,S,1,pairs,2,2] = [[cos, -sin], [sin, cos]]
    a, b = hidden_states[..., :pairs].float(), hidden_states[..., pairs:2 * pairs].float()
    rotated = torch.cat((r[..., 0, 0] * a + r[..., 0, 1] * b, r[..., 1, 0] * a + r[..., 1, 1] * b), dim=-1)
    return torch.cat((rotated.to(hidden_states.dtype), hidden_states[..., 2 * pairs:]), dim=-1)


_orig_mod_scale_shift, _orig_mod_gate = h3model._mod_scale_shift, h3model._mod_gate   # kept for tests
_row_index = {}


def _rows(batch_segments, device):
    """Per-row modulation index of the (single-sample) segment plan, built once per forward: every block
    of a forward call shares the same plan object."""
    key = (id(batch_segments), device)
    hit = _row_index.get(key)
    if hit is not None and hit[0] is batch_segments:
        return hit[1]
    (segments,) = batch_segments
    seg = torch.tensor(segments, dtype=torch.long)
    idx = torch.repeat_interleave(seg[:, 2], seg[:, 1] - seg[:, 0]).to(device, non_blocking=True)
    _row_index.clear()
    _row_index[key] = (batch_segments, idx)
    return idx


def mod_scale_shift(hidden_states, shift, scale, batch_segments):
    """musubi's _mod_scale_shift without the Python loop over segments (hundreds of small kernels per block
    with the text's token tags): the same x * (1 + scale) + shift, gathered per row."""
    idx = _rows(batch_segments, hidden_states.device)
    return hidden_states * (1.0 + scale[idx]).unsqueeze(0) + shift[idx].unsqueeze(0)


def mod_gate(residual, update, gate, batch_segments):
    """musubi's _mod_gate, vectorised the same way: residual + update * gate."""
    idx = _rows(batch_segments, residual.device)
    return residual + update * gate[idx].unsqueeze(0)


def install(model) -> None:
    """Patch H3 attention, its RoPE and its AdaLN modulation (idempotent) and number the layers for the KV
    cache."""
    h3model.Attention.forward = _forward
    h3model._apply_rope_split_half = rope                      # the released (unmasked) forward too
    h3model._mod_scale_shift, h3model._mod_gate = mod_scale_shift, mod_gate
    for i, blk in enumerate(model.blocks):
        (blk.attn if hasattr(blk, "attn") else blk.module.attn)._rtcwm_layer = i    # FSDP-wrapped or not


@contextlib.contextmanager
def active(mask=None, kv=None):
    _state["mask"], _state["kv"] = mask, kv
    try:
        yield
    finally:
        _state["mask"], _state["kv"] = None, None
