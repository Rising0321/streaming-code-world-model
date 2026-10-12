"""Training samples: spans of 6 consecutive chunks cut from a clip's latent stream, each with a start frame.

A clip's stream (rtcwm.train.preencode --stage stream) holds K chunks (17-frame VAE clips, K = 54 for the
934-frame renders). The span starting at chunk j covers frames 17j .. 17j+101 and is packed with
    x / DUV   : the stream's latents 5j .. 5j+29
    Picture 1 : a start frame m <= j (rtcwm.data.anchors: the first frame of every 3rd chunk; native
                1344x768 by default), with its static caption's Qwen states (that frame only, no timing)
    audio     : silent
With probability P_START the span starts at its start frame (j = m: generation from that image, no
history; stream_start, so the static rows see chunk 0); otherwise the start frame is an earlier one
(the stream has been running since m). The sample list is every span start of the kept clips
(data_lists/clips.json: no Town02 / Town04) minus spans overlapping a caption window flagged by QC
(data_lists/bad_windows.json). A fixed set of clips is held out.
"""
from __future__ import annotations

import random

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from cwm_h3_inference.constants import AUDIO_LATENT_SHAPE, STRIDE_FRAMES, VIDEO_LATENT_SHAPE, WINDOW_FRAMES

from .. import paths
from ..data.anchors import cache_paths, clip_anchors
from ..data.qc import lists
from ..model.layout import CHUNK_LATENTS, CLIP_FRAMES, TARGET_LATENTS, SPAN_CHUNKS
from ..model.packing import audio_frames

P_START = 0.2


def stream_chunks(clip: str) -> int:
    with safe_open(str(paths.CACHE / clip / "stream.safetensors"), "pt") as f:
        return f.get_slice("duv").get_shape()[1] // CHUNK_LATENTS


def start_frames(clip: str, res: str = "native") -> list[int]:
    """Start chunks of a clip whose Picture 1 and caption states are encoded (at Picture 1 resolution res)."""
    return [m for m in clip_anchors(clip)
            if cache_paths(clip, m, res)["text"].exists() and cache_paths(clip, m, res)["latent"].exists()]


def list_samples(val_clips: int, seed: int, res: str = "native"):
    """(train, val, n_dropped) as lists of (clip, span start chunk)."""
    keep, bad = lists.load()
    samples, dropped = [], 0
    for c in sorted(keep):
        if not (paths.CACHE / c / "stream.safetensors").exists() or 0 not in start_frames(c, res):
            continue
        flagged = [(STRIDE_FRAMES * w, STRIDE_FRAMES * w + WINDOW_FRAMES) for w in bad.get(c, ())]
        for j in range(stream_chunks(c) - SPAN_CHUNKS + 1):
            a, b = CLIP_FRAMES * j, CLIP_FRAMES * (j + SPAN_CHUNKS)
            if any(a < fb and fa < b for fa, fb in flagged):
                dropped += 1
                continue
            samples.append((c, j))
    names = sorted({c for c, _ in samples})
    random.Random(seed).shuffle(names)
    val = set(names[:val_clips])
    return [s for s in samples if s[0] not in val], [s for s in samples if s[0] in val], dropped


class StartFrames:
    """Draws (span start j, start frame m) for a listed (clip, j): with probability P_START a span that
    starts at one of the clip's start frames, otherwise the listed span with an earlier start frame."""

    def __init__(self, samples, seed: int, res: str = "native"):
        self.rng = random.Random(seed)
        self.spans = {}
        for c, j in samples:
            self.spans.setdefault(c, set()).add(j)
        self.frames = {c: start_frames(c, res) for c in self.spans}

    def draw(self, clip: str, j: int) -> tuple[int, int]:
        starts = [m for m in self.frames[clip] if m in self.spans[clip]]
        if starts and self.rng.random() < P_START:
            m = self.rng.choice(starts)
            return m, m
        earlier = [m for m in self.frames[clip] if m < j]
        if earlier:
            return j, self.rng.choice(earlier)
        m = max(m for m in self.frames[clip] if m <= j)          # j = 0, or no start frame before it
        return m, m


def silence() -> torch.Tensor:
    a = load_file(paths.CACHE / "silence_audio.safetensors")["audio"]
    assert tuple(a.shape) == AUDIO_LATENT_SHAPE[1:]
    return a


def load_window(clip: str, j: int, silence_audio: torch.Tensor, start: int = 0, with_target: bool = True,
                res: str = "native", n_latents: int = TARGET_LATENTS) -> dict:
    """The span starting at stream chunk j, with start frame (chunk) start <= j: x [1,24,30,48,84]
    (with_target), DUV ref [1,24,30,12,22], Picture 1, silent audio, Qwen states, stream_start.
    n_latents = 37: the released window over the same frames (124 frames from chunk j; the stream's per-clip
    latents are exactly its encoding, the VAE being causal inside a clip), with the released audio length."""
    assert start <= j, (start, j)
    d = paths.CACHE / clip
    lat = slice(CHUNK_LATENTS * j, CHUNK_LATENTS * j + n_latents)
    with safe_open(str(d / "stream.safetensors"), "pt") as f:
        duv = f.get_slice("duv")[:, lat]
        x = f.get_slice("target")[:, lat].unsqueeze(0) if with_target else None
    assert duv.shape[1] == n_latents, (clip, j, duv.shape)
    cp = cache_paths(clip, start, res)
    t = load_file(cp["text"])
    anchor = load_file(cp["latent"])["latents_ref_000_image"].unsqueeze(0)
    ref = torch.nn.functional.pad(duv.unsqueeze(0), (0, 1, 0, 0), value=0.0)          # to the runtime width
    n_audio = AUDIO_LATENT_SHAPE[-1] if n_latents == VIDEO_LATENT_SHAPE[2] else audio_frames(n_latents)
    audio = silence_audio[..., :n_audio].unsqueeze(0).clone()
    out = dict(audio=audio, anchor=anchor, ref=ref, hidden=t["hidden"].unsqueeze(0), tags=t["tags"].unsqueeze(0),
               stream_start=j == start)
    if with_target:
        out["x"] = x
    return out
