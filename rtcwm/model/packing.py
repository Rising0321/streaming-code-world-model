"""Packing one window's condition and audio rows, and running the H3 blocks on a packed sequence."""
from __future__ import annotations

import torch

from musubi_tuner.minimax_h3.packing import (H3VideoGeometry, _audio_grid, _build_modulation_segments, _frame_grid,
                                             _unpack_video_rows, _video_grid, _video_t_spans, pack_audio_rows,
                                             pack_video_rows)

from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

from . import attention
from .layout import CLEAN_T, CLIP_FRAMES, FRAME_ROWS, LATENT_SHAPE, TARGET_LATENTS, TEXT_T_FIXED, VISUAL_T


def audio_frames(n_latents: int) -> int:
    """Audio latent frames that cover n_latents video latents (whole VAE clips), as the released layout counts them."""
    return (10 * (n_latents // 5) * CLIP_FRAMES + 3) // 6


def window_positions(text_len: int, anchor_hw, duv_geometry: H3VideoGeometry, n_audio: int, n_latents: int):
    """(text, Picture 1, DUV, audio, video) positions of a window, laid out exactly as musubi's
    build_position_grid lays out a Ref2VA window (tests/test_positions.py), for any number of latents:
    one time axis -- text, Picture 1, the DUV video, then the audio and the target video."""
    text = torch.zeros(text_len, 3, dtype=torch.float64)
    text[:, 0] = torch.arange(text_len, dtype=torch.float64)
    cursor = float(text_len)
    frame, _ = _frame_grid(H3VideoGeometry(1, *anchor_hw))
    image = torch.empty(frame.shape[0], 3, dtype=torch.float64)
    image[:, 0], image[:, 1:] = cursor, frame
    cursor += 1.0
    duv = _video_grid(duv_geometry, cursor)
    cursor += sum(_video_t_spans(duv_geometry.frames))
    target = H3VideoGeometry(n_latents, *LATENT_SHAPE[1:])
    _, width = _frame_grid(target)
    audio = _audio_grid(cursor, n_audio, float(width[0]), float(width[-1]))
    return text, image, duv, audio, _video_grid(target, cursor)


class Window:
    """Projected [text | Picture 1 | DUV | audio] rows of one span and the positions of its video latents.

    Condition times are fixed (text 1, Picture 1 / DUV 0.999, audio 1), so the condition rows' keys and
    values do not depend on the denoising step. static_sees_first: the span starts the stream, and its
    static rows see chunk 0 (rtcwm.model.masks)."""

    def __init__(self, model, *, text_hidden, text_tags, anchor, duv, audio, n_latents: int = TARGET_LATENTS,
                 static_sees_first: bool = True):
        dev = model.video_patch_proj.weight.device
        self.L = text_hidden.shape[1]
        text_pos, image_pos, duv_pos, audio_pos, video_pos = window_positions(
            self.L, anchor.shape[3:], H3VideoGeometry(*duv.shape[2:]), audio.shape[-1], n_latents)
        self.ca_pos = torch.cat([text_pos, image_pos, duv_pos, audio_pos])
        self.target_pos = video_pos
        vis = [pack_video_rows(t.to(dev)) for t in (anchor, duv)]
        self.n_visual = vis[0].shape[1] + vis[1].shape[1]          # Picture 1 + DUV rows
        self.head_rows, self.duv_rows = self.L + vis[0].shape[1], vis[1].shape[1]
        self.static_sees_first = static_sees_first
        assert image_pos.shape[0] == vis[0].shape[1] and duv_pos.shape[0] == vis[1].shape[1]
        pv = model.video_patch_proj(torch.cat(vis, 1).to(torch.float32)).to(model.dtype)
        pa = model.audio_patch_proj(pack_audio_rows(audio.to(dev)).to(torch.float32)).to(model.dtype)
        text = model.token_refiner(model.condition_proj(text_hidden.to(dev, model.dtype)))
        self.rows = torch.cat([text, pv, pa], 1)
        self.cond_rows, self.audio_rows = self.L + pv.shape[1], pa.shape[1]
        assert self.rows.shape[1] == self.ca_pos.shape[0]
        tags = torch.zeros(self.rows.shape[1], dtype=torch.int64)
        tags[:self.L] = text_tags[0].cpu()
        tags[self.cond_rows:] = 2
        self.tags, self.dev = tags, dev

    def cond_times(self) -> torch.Tensor:
        t = torch.full((self.rows.shape[1],), CLEAN_T)
        t[:self.L] = TEXT_T_FIXED
        t[self.L:self.L + self.n_visual] = VISUAL_T
        return t

    def video_rows(self, model, latents):
        return model.video_patch_proj(pack_video_rows(latents.to(self.dev)).to(torch.float32)).to(model.dtype)


def run_blocks(model, hidden, t_rows, tags, positions, *, mask=None, kv=None, train: bool = False,
               offload: bool = False):
    """All H3 blocks over a packed sequence with per-row times. Returns (hidden, time embeddings, row ->
    unique-time index). train: activation checkpointing per block; offload: checkpointed block inputs kept on
    the CPU."""
    uniq, inv = torch.unique(t_rows, sorted=True, return_inverse=True)
    segments = (_build_modulation_segments(3 * inv + tags),)
    temb = model.time_embedder(uniq.to(hidden.device)).to(model.dtype)
    rot = model._rotation_table(positions.unsqueeze(0).to(hidden.device), hidden.dtype)

    def wrap(block):
        def run(*args):
            with attention.active(mask, kv):
                return block(*args)
        return run

    dev = hidden.device
    for block in model.blocks:
        fn = wrap(block)
        if train:
            if offload:
                fn = create_cpu_offloading_wrapper(fn, dev)
            hidden = torch.utils.checkpoint.checkpoint(fn, hidden, temb, segments, rot, use_reentrant=False)
        else:
            hidden = fn(hidden, temb, segments, rot)
    return hidden.to(dev), temb, inv          # the offloading wrapper returns the last output on the CPU


def video_output(model, hidden, temb, inv, start: int, n_latents: int):
    """Final layer on rows [start, start + n_latents frames) -> velocity [1,24,n_latents,48,84]."""
    rows = slice(start, start + n_latents * FRAME_ROWS)
    empty = slice(start, start)
    vid, _ = model.final_layer(hidden, temb, video_slice=rows, audio_slice=empty,
                               video_timestep_index=int(inv[start]), audio_timestep_index=int(inv[start]),
                               video_timestep_indices=inv[rows].unsqueeze(0), audio_timestep_indices=inv[empty].unsqueeze(0))
    return _unpack_video_rows(vid, H3VideoGeometry(n_latents, *LATENT_SHAPE[1:]))
