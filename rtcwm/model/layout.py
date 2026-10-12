"""Streaming chunk layout and the per-row bookkeeping shared by the objective and the sampler.

A clip is one continuous latent stream (rtcwm.train.preencode --stage stream). The H3 video VAE encodes
independent 17-frame clips into 5 latents each; one chunk is one such clip (17 frames, 0.71 s), so a chunk
never splits a VAE clip and every generated prefix ends on a clip boundary. There is no keyframe: chunk k
is generated from the conditions and the chunks before it in its attention span, which slides with k
(there is no generation window and no restart anywhere in the stream). The span (default 6 chunks: 30
latents, 4.25 s; RTCWM_SPAN_CHUNKS sets it, for training and sampling alike -- a LoRA records the span it
was trained with) is packed as a standalone H3 sequence with positions counted from the span's first chunk
(rtcwm.model.packing.Window), so a late chunk never leaves the position range of the released model.
"""
from __future__ import annotations

import os
from itertools import accumulate

CHUNK_LATENTS = 5                    # one 17-frame VAE clip
CLIP_FRAMES = 17
SPAN_CHUNKS = int(os.environ.get("RTCWM_SPAN_CHUNKS", "6"))   # attention span: a chunk sees itself and the ones before it
CHUNK_SIZES = (CHUNK_LATENTS,) * SPAN_CHUNKS
NUM_CHUNKS = SPAN_CHUNKS             # chunks in one packed span
TARGET_LATENTS = sum(CHUNK_SIZES)    # video latents per packed span (30 for 6 chunks)
CHUNK_STARTS = tuple(accumulate((0,) + CHUNK_SIZES[:-1]))
FRAME_ROWS = 1008                    # (48/2) * (84/2) packed rows per latent frame
LATENT_SHAPE = (24, 48, 84)          # channels, height, width

# condition row times (model time: 1 = clean)
TEXT_T_FIXED, VISUAL_T, CLEAN_T = 1.0, 0.999, 1.0


def chunk_latents(c: int) -> slice:
    """Latents of chunk c of a packed span."""
    return slice(CHUNK_STARTS[c], CHUNK_STARTS[c] + CHUNK_SIZES[c])


def chunk_rows(c: int) -> int:
    return CHUNK_SIZES[c] * FRAME_ROWS


def chunk_row_slice(c: int) -> slice:
    """Packed rows of chunk c within the span's video rows."""
    return slice(CHUNK_STARTS[c] * FRAME_ROWS, (CHUNK_STARTS[c] + CHUNK_SIZES[c]) * FRAME_ROWS)


def chunk_shape(c: int = 0) -> tuple:
    return (1, LATENT_SHAPE[0], CHUNK_SIZES[c], *LATENT_SHAPE[1:])


def span_start(k: int) -> int:
    """First chunk of the attention span of stream chunk k."""
    return max(0, k - (SPAN_CHUNKS - 1))
