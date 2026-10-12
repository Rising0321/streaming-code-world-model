"""Block-causal flex-attention mask over a packed span.

Rows come in contiguous runs of one (role, chunk). The BlockMask's full / partial 128x128 tiles are
derived from those few runs (never a token-level [S,S] mask; the builder follows SolarWM's
backends/minimax_h3/torch_mask.py, Apache-2.0), and mask_mod decides single tokens from a handful of
captured scalars, so the compiled kernel does not specialise on the caption length.

A span is packed [text | Picture 1 | DUV chunks 0..5 | audio | video chunks first .. first+n-1]. Roles:
STATIC (text, Picture 1, audio), DUV (the coarse proxy, one run per chunk) and VIDEO. Nothing sees a
proxy chunk later than its own (real-time control: the future proxy does not exist at inference).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask

from .layout import CHUNK_SIZES, FRAME_ROWS, NUM_CHUNKS, SPAN_CHUNKS

STATIC, DUV, VIDEO = 0, 1, 2
TILE = 128
assert len(set(CHUNK_SIZES)) == 1
CHUNK_ROWS = CHUNK_SIZES[0] * FRAME_ROWS


def df_allows(qr, qc, kr, kc, static_sees_first: bool) -> bool:
    """Diffusion forcing with a causal proxy.
        static    : static rows, and -- static_sees_first, i.e. the span starts the stream -- proxy and video
                    chunk 0 (with no history, chunk 0 needs condition rows that see it, as in the released
                    model; with a history, conditions that see only themselves do better)
        proxy p   : static rows and proxy chunks <= p
        video c   : static rows, proxy chunks <= c and video chunks c-5..c"""
    if qr == STATIC:
        return kr == STATIC or (static_sees_first and kc == 0 and kr in (DUV, VIDEO))
    if qr == DUV:
        return kr == STATIC or (kr == DUV and kc <= qc)
    return kr == STATIC or (kr == DUV and kc <= qc) or (kr == VIDEO and 0 <= qc - kc < SPAN_CHUNKS)


def _tiles(runs, allows, S, device):
    allowed = {(r, c): [(a, b) for a, b, r2, c2 in runs if allows(r, c, r2, c2)] for _, _, r, c in runs}
    nb = math.ceil(S / TILE)
    ks = torch.arange(nb) * TILE
    ke = torch.clamp(ks + TILE, max=S)
    full = torch.zeros(nb, nb, dtype=torch.bool)
    anyt = torch.zeros(nb, nb, dtype=torch.bool)
    for qb in range(nb):
        q0, q1 = qb * TILE, min(qb * TILE + TILE, S)
        rf, ra = torch.ones(nb, dtype=torch.bool), torch.zeros(nb, dtype=torch.bool)
        for a, b, r, c in runs:
            if a < q1 and q0 < b:
                sf, sa = torch.zeros(nb, dtype=torch.bool), torch.zeros(nb, dtype=torch.bool)
                for a2, b2 in allowed[(r, c)]:
                    sf |= (a2 <= ks) & (ke <= b2)
                    sa |= (ks < b2) & (a2 < ke)
                rf &= sf
                ra |= sa
        full[qb], anyt[qb] = rf, ra
    mb = 1 << (nb - 1).bit_length()

    def meta(m):
        o = torch.zeros(mb, mb, dtype=torch.int32)
        o[:nb, :nb] = m
        return (o.sum(-1).to(torch.int32)[None, None].to(device),
                torch.argsort(o, dim=-1, descending=True, stable=True).to(torch.int32)[None, None].to(device))

    pc, pi = meta(anyt & ~full)
    fc, fi = meta(full)
    return pc, pi, fc, fi


def _scalars(device, *values):
    return [torch.tensor(v, dtype=torch.long, device=device) for v in values]


def span_runs(head_rows: int, duv_rows: int, audio_rows: int, n_chunks: int, first_chunk: int = 0):
    """(runs, total rows) of [head | DUV x6 | audio | video chunks first..]; runs are (start, end, role, chunk)."""
    assert duv_rows % NUM_CHUNKS == 0, duv_rows
    dr = duv_rows // NUM_CHUNKS
    runs = [(0, head_rows, STATIC, -1)]
    runs += [(head_rows + p * dr, head_rows + (p + 1) * dr, DUV, p) for p in range(NUM_CHUNKS)]
    v0 = head_rows + duv_rows + audio_rows
    runs.append((head_rows + duv_rows, v0, STATIC, -1))
    runs += [(v0 + j * CHUNK_ROWS, v0 + (j + 1) * CHUNK_ROWS, VIDEO, first_chunk + j) for j in range(n_chunks)]
    return [r for r in runs if r[1] > r[0]], v0 + n_chunks * CHUNK_ROWS


def df_mask(head_rows: int, duv_rows: int, audio_rows: int, n_chunks: int, device, first_chunk: int = 0,
            static_sees_first: bool = True) -> BlockMask:
    """Mask of a packed span [text | Picture 1 | DUV chunks 0..5 | audio | video chunks first ..]."""
    runs, S = span_runs(head_rows, duv_rows, audio_rows, n_chunks, first_chunk)
    pc, pi, fc, fi = _tiles(runs, lambda *a: df_allows(*a, static_sees_first), S, device)
    d0, d1, v0, se, dr, cr, wb, first, ssf = _scalars(
        device, head_rows, head_rows + duv_rows, head_rows + duv_rows + audio_rows, S, duv_rows // NUM_CHUNKS,
        CHUNK_ROWS, SPAN_CHUNKS, first_chunk, int(static_sees_first))

    def classify(i):
        duv = (i >= d0) & (i < d1)
        vid = (i >= v0) & (i < se)
        chunk = torch.where(duv, torch.div(i - d0, dr, rounding_mode="floor"),
                            torch.div(i - v0, cr, rounding_mode="floor") + first)
        return ~duv & ~vid, duv, vid, chunk

    def mask_mod(_b, _h, qi, ki):
        q_st, q_du, q_vi, qc = classify(qi)
        k_st, k_du, k_vi, kc = classify(ki)
        first_rows = (ssf == 1) & (kc == 0) & (k_du | k_vi)
        return ((q_st & (k_st | first_rows)) | (q_du & (k_st | (k_du & (kc <= qc))))
                | (q_vi & (k_st | (k_du & (kc <= qc)) | (k_vi & (qc - kc >= 0) & (qc - kc < wb)))))

    return BlockMask.from_kv_blocks(pc, pi, fc, fi, BLOCK_SIZE=TILE, mask_mod=mask_mod, seq_lengths=(S, S))


@dataclass(frozen=True)
class DenseDF:
    """The df_mask of a training span (video chunks 0..n-1, n <= SPAN_CHUNKS) for dense FlashAttention.

    Under the rule every query run's keys are one contiguous prefix of one of two key orders:
        main  [static | DUV 0 | video 0 | DUV 1 | video 1 | ...]   static rows, video chunk c
        proxy [static | DUV 0 | DUV 1 | ...]                        proxy chunk p
    (static = text, Picture 1 and audio), so the masked attention is a few unmasked prefix calls with the
    same result as the BlockMask, without its per-tile mask evaluation (rtcwm.model.attention)."""
    head_rows: int
    duv_rows: int
    audio_rows: int
    n_chunks: int
    static_sees_first: bool = True

    def __post_init__(self):
        assert 0 < self.n_chunks <= SPAN_CHUNKS and self.duv_rows % NUM_CHUNKS == 0, self
