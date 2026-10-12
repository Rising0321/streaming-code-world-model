"""The causal training objective (diffusion forcing) on one packed window.

CWM's flow-matching parameterisation: noisy = (1 - sigma) x + sigma n, model time 1 - sigma, target
velocity x - n, sigma flow-shifted with CWM's video shift.
"""
from __future__ import annotations

import torch

from cwm_h3_inference.constants import VIDEO_SHIFT, VISUAL_CONDITION_CLEAN

from .layout import CHUNK_SIZES, FRAME_ROWS, NUM_CHUNKS, TARGET_LATENTS, chunk_rows
from .masks import DenseDF, df_mask
from .packing import Window, run_blocks, video_output

OBJECTIVES = ("df",)


def shift(base, s: float = VIDEO_SHIFT):
    """Flow-shifted noise amount, as in musubi's H3 trainer: s*t / (1 + (s-1)*t)."""
    return s * base / (1.0 + (s - 1.0) * base)


def _cached(cache: dict, key, build):
    if key not in cache:
        cache.clear()
        cache[key] = build()
    return cache[key]


def df_forward(model, win: Window, video, chunk_t, mask_cache: dict, train: bool = True, offload: bool = False,
               attn: str = "fa3"):
    """video [1,24,30,H,W], each chunk noised to its level; chunk_t: 6 model times (1 = clean).
    attn: fa3 (dense prefix calls) or flex (BlockMask), the same mask. Returns the predicted velocity of
    all 30 latents."""
    hidden = torch.cat([win.rows, win.video_rows(model, video)], 1)
    rows = torch.tensor([chunk_rows(c) for c in range(NUM_CHUNKS)])
    t_rows = torch.cat([win.cond_times(), torch.tensor(chunk_t, dtype=torch.float32).repeat_interleave(rows)])
    tags = torch.cat([win.tags, torch.zeros(TARGET_LATENTS * FRAME_ROWS, dtype=torch.int64)])
    if attn == "fa3":
        mask = DenseDF(win.head_rows, win.duv_rows, win.audio_rows, NUM_CHUNKS, win.static_sees_first)
    else:
        key = ("df", win.head_rows, win.duv_rows, win.audio_rows, win.static_sees_first)
        mask = _cached(mask_cache, key, lambda: df_mask(win.head_rows, win.duv_rows, win.audio_rows, NUM_CHUNKS, win.dev,
                                                        static_sees_first=win.static_sees_first))
    hidden, temb, inv = run_blocks(model, hidden, t_rows, tags, torch.cat([win.ca_pos, win.target_pos]),
                                   mask=mask, train=train, offload=offload)
    return video_output(model, hidden, temb, inv, win.rows.shape[1], TARGET_LATENTS)


def augment_conditions(sample, dev, gen):
    """Picture 1 and the DUV reference mixed 0.999 clean / 0.001 noise, as at CWM inference."""
    ref, anc = sample["ref"].to(dev), sample["anchor"].to(dev)
    ref = VISUAL_CONDITION_CLEAN * ref + (1 - VISUAL_CONDITION_CLEAN) * torch.randn(ref.shape, generator=gen).to(dev)
    anc = VISUAL_CONDITION_CLEAN * anc + (1 - VISUAL_CONDITION_CLEAN) * torch.randn(anc.shape, generator=gen).to(dev)
    return anc, ref


def make_window(model, sample, dev, gen) -> Window:
    anc, ref = augment_conditions(sample, dev, gen)
    return Window(model, text_hidden=sample["hidden"], text_tags=sample["tags"], anchor=anc, duv=ref,
                  audio=sample["audio"].to(dev), n_latents=sample["ref"].shape[2],
                  static_sees_first=bool(sample.get("stream_start", True)))


def df_levels(gen, level_shift: float = 3.0) -> torch.Tensor:
    """Per-chunk noise amounts, independent per chunk and flow-shifted by level_shift. History chunks are never
    the clean ground truth: a clean ground-truth history teaches the model to trust its history completely,
    which it must not do with its own (imperfect) history at inference (v3 drew one with probability 0.5 and
    drifted more); at a moderate shift the history comes at every noise level and the model learns to use it
    with care."""
    return shift(torch.rand(NUM_CHUNKS, generator=gen), level_shift)


def df_loss(model, sample, dev, gen, mask_cache, offload: bool = False, attn: str = "fa3",
            level_shift: float = 3.0):
    x = sample["x"].to(dev)
    sig = df_levels(gen, level_shift)
    s_lat = sig.repeat_interleave(torch.tensor(CHUNK_SIZES)).to(dev).view(1, 1, -1, 1, 1)
    n = torch.randn(x.shape, generator=gen).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        win = make_window(model, sample, dev, gen)
        pred = df_forward(model, win, (1 - s_lat) * x + s_lat * n, [float(1.0 - v) for v in sig], mask_cache,
                          offload=offload, attn=attn)
    noisy = (s_lat > 0).float()
    se = ((pred.float() - (x - n).float()) ** 2) * noisy
    return se.sum() / (noisy.sum() * x.shape[1] * x.shape[3] * x.shape[4]).clamp_min(1.0)


LOSSES = dict(df=df_loss)
