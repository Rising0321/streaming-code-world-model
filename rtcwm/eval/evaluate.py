"""Stream held-out clips chunk by chunk and compare with the CARLA ground truth.

    python -m rtcwm.eval.evaluate --lora LORA --out DIR --clips clip_0244 [...] [--chunks N]
           [--steps 35] [--rule euler|x0] [--shift S]          (distilled students: --steps 4 --rule x0 --shift 5)

Each clip is generated from nothing (no keyframe) as one stream of --chunks 17-frame chunks (default: all
54, 918 frames / 38 s), every chunk in its window of the 5 chunks before it (rtcwm.model.sampler).
Writes DIR/<clip>_stream_gen_vs_gt.mp4 (generated | GT, full resolution), DIR/<clip>_latents.pt and DIR/<clip>_stream.json:
pixel MSE overall and per chunk, the luma jump at chunk boundaries vs inside chunks, seconds per chunk.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cwm_h3_inference.constants import VIDEO_SHIFT

from .. import paths
from ..model.layout import CLIP_FRAMES
from ..model.loading import load_for_inference
from ..model.sampler import generate_stream
from ..train.dataset import load_window, silence, stream_chunks
from .video import decode, load_vae, read_mp4, write_mp4


def luma(frames: np.ndarray) -> np.ndarray:
    f = frames.astype(np.float32)
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).mean(axis=(1, 2))


def report(gen: np.ndarray, gt: np.ndarray) -> dict:
    err = ((gen.astype(np.float32) - gt.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
    jumps, gt_jumps = np.abs(np.diff(luma(gen))), np.abs(np.diff(luma(gt)))
    boundary = np.zeros(len(jumps), bool)
    boundary[CLIP_FRAMES - 1::CLIP_FRAMES] = True                  # frame 17k-1 -> 17k
    n = len(gen) // CLIP_FRAMES
    return dict(frames=int(len(gen)), pixel_mse=float(err.mean()),
                chunk_mse=[float(err[CLIP_FRAMES * k:CLIP_FRAMES * (k + 1)].mean()) for k in range(n)],
                boundary_luma_jump=float(jumps[boundary].mean()), inner_luma_jump=float(jumps[~boundary].mean()),
                gt_boundary_luma_jump=float(gt_jumps[boundary].mean()), max_luma_jump=float(jumps.max()))


def check_span(lora):
    """The LoRA's training span (its metadata; 6 if it predates the setting) must be the sampler's."""
    from safetensors import safe_open
    from ..model.layout import SPAN_CHUNKS
    with safe_open(str(lora), "pt") as f:
        trained = int((f.metadata() or {}).get("span_chunks", "6"))
    assert trained == SPAN_CHUNKS, f"{lora} was trained with span {trained}; set RTCWM_SPAN_CHUNKS={trained}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", type=Path, default=paths.CWM_LORA)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--chunks", type=int, default=0, help="0 = the whole stream")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--rule", choices=["euler", "x0"], default="euler", help="x0: few-step (distilled) sampling")
    ap.add_argument("--shift", type=float, default=VIDEO_SHIFT, help="flow shift of the sampling schedule")
    ap.add_argument("--history-noise", type=float, default=0.0, help="noise level of the history chunks")
    ap.add_argument("--picture1", choices=["native", "2048"], default="native", help="Picture 1 resolution the LoRA was trained with")
    ap.add_argument("--kv-reuse", action="store_true", help="carry the history's keys/values across chunks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp8", action="store_true", help="the blocks' large linears in FP8 (Hopper)")
    ap.add_argument("--history-from-last-step", action=argparse.BooleanOptionalAction, default=True,
                    help="kv-reuse: cache each chunk from its last denoising step (one forward fewer per chunk; "
                         "--no-history-from-last-step: one more clean forward)")
    a = ap.parse_args()
    dev = torch.device("cuda", 0)
    a.out.mkdir(parents=True, exist_ok=True)
    check_span(a.lora)
    from ..model import sampler
    sampler.HISTORY_FROM_LAST_STEP = a.history_from_last_step
    model = load_for_inference(a.lora, dev, fp8=a.fp8)
    audio, results = silence(), {}
    for clip in a.clips:
        n = a.chunks or stream_chunks(clip)
        secs = []
        t0 = time.time()
        lat = generate_stream(model, lambda j: load_window(clip, j, audio, start=0, with_target=False, res=a.picture1), n, a.steps,
                              torch.Generator().manual_seed(a.seed), dev, a.rule, a.shift, a.history_noise, a.kv_reuse,
                              on_chunk=lambda k, s: (secs.append(round(s, 1)),
                                                     print(f"{clip} chunk {k + 1}/{n} {s:.1f}s", flush=True)))
        results[clip] = (lat.cpu(), secs, time.time() - t0)
    del model
    torch.cuda.empty_cache()
    vae = load_vae(dev)
    for clip, (lat, secs, total) in results.items():
        gen = decode(vae, lat, dev)
        gt = read_mp4(paths.DATASET / clip / "rgb.mp4")[:len(gen)]
        rep = dict(clip=clip, lora=str(a.lora), steps=a.steps, rule=a.rule, history_noise=a.history_noise, history_from_last_step=a.history_from_last_step, kv_reuse=a.kv_reuse, chunks=len(secs), **report(gen, gt),
                   seconds_per_chunk=secs, gen_seconds=round(total, 1))
        (a.out / f"{clip}_stream.json").write_text(json.dumps(rep, indent=1))
        torch.save(lat, a.out / f"{clip}_latents.pt")
        write_mp4(a.out / f"{clip}_stream_gen_vs_gt.mp4", np.concatenate([gen, gt], 2), crf=20)
        print(json.dumps({k: v for k, v in rep.items() if k not in ("chunk_mse", "seconds_per_chunk")}), flush=True)


if __name__ == "__main__":
    main()
