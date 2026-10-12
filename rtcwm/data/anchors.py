"""Start frames (Picture 1) for streaming generation.

A stream starts from a start frame: Picture 1, its static caption and the proxy of that frame. Training
draws Picture 1 from many points of a clip -- the first frame of every ANCHOR_EVERY-th chunk -- either
right at the start of the sampled span (generation from that image, no history) or earlier in the clip.
Start frames inside a caption window flagged by QC are skipped. Per start chunk m:
    CLIP/prompts/static_cMM.txt                      the caption (rtcwm.data.captions --static)
    CACHE/CLIP/anchor_cMM.png                        the frame
    CACHE/CLIP/anchor[n]_cMM.safetensors             its Picture 1 latent (n: native 1344x768, else 2048 short edge)
    CACHE/CLIP/static[n]_cMM.text.safetensors        its Qwen states (rtcwm.train.preencode --stage static_text)
"""
from __future__ import annotations

from pathlib import Path

from cwm_h3_inference.constants import STRIDE_FRAMES, WINDOW_FRAMES

from .. import paths
from .clip import read_meta

ANCHOR_EVERY = 3                    # chunks between start frames
CLIP_FRAMES = 17                    # frames per chunk (one VAE clip)
STREAM_CHUNKS = 54                  # chunks of a 934-frame render's stream
LAST_SPAN_START = STREAM_CHUNKS - 6


def anchor_frame(m: int) -> int:
    return CLIP_FRAMES * m


def _flagged(clip: str):
    from ..data.qc import lists
    _, bad = lists.load()
    return [(STRIDE_FRAMES * w, STRIDE_FRAMES * w + WINDOW_FRAMES) for w in bad.get(clip, ())]


def clip_anchors(clip: Path | str) -> list[int]:
    """Start chunks of a clip: every ANCHOR_EVERY-th chunk up to the last span start, outside flagged windows."""
    name = clip if isinstance(clip, str) else clip.name
    flagged = _flagged(name)
    return [m for m in range(0, LAST_SPAN_START + 1, ANCHOR_EVERY)
            if not any(a <= anchor_frame(m) < b for a, b in flagged)]


def caption_path(clip: Path, m: int) -> Path:
    return clip / "prompts" / f"static_c{m:02d}.txt"


# Picture 1 resolution. "native": the renders' 1344x768 frame as is (1008 H3 rows, ~2.6k Qwen rows; equal or
# better than the released 2048 short edge, which only upsamples our frames -- ops/dbg/res_test.py).
# "2048": the released CWM anchor size (7168 H3 rows, ~8.8k Qwen rows), used by stream_df_v1 / v2.
PICTURE1_RES = ("native", "2048")


def cache_paths(clip: str, m: int, res: str = "native") -> dict:
    assert res in PICTURE1_RES, res
    d = paths.CACHE / clip
    tag = "n" if res == "native" else ""
    return dict(png=d / f"anchor_c{m:02d}.png", latent=d / f"anchor{tag}_c{m:02d}.safetensors",
                text=d / f"static{tag}_c{m:02d}.text.safetensors")
