"""KV reuse (GPU): keys moved by a rotation inside attention (one shared cached copy, rtcwm.model.sampler.
_moved) == keys re-rotated into a moved copy at every span move (the earlier behaviour), over a stream long
enough for the span to move four times; both bf16, so up to rounding (one rotation instead of a chain of
them). Also reports the time of each.

    python tests/test_kv_reuse.py
"""
import time

import torch

from rtcwm import paths
from rtcwm.model import sampler
from rtcwm.model.loading import load_for_inference
from rtcwm.train.dataset import load_window, silence

TOL = 0.02
N_CHUNKS = 10


def test_kv_reuse(clip="clip_0244"):
    dev = torch.device("cuda", 0)
    model = load_for_inference(paths.RUNS / "stream_df_v3/lora_step00300.safetensors", dev)
    audio = silence()
    out = {}
    for rotate in (False, True, False, True):
        sampler.ROTATE_IN_ATTENTION = rotate
        torch.cuda.synchronize(); t0 = time.time()
        x = sampler.generate_stream(model, lambda j: load_window(clip, j, audio, start=0, with_target=False), N_CHUNKS, 4,
                                    torch.Generator().manual_seed(0), dev, "x0", 5.0, kv_reuse=True).float()
        torch.cuda.synchronize()
        print(f"rotate_in_attention={rotate}: {time.time() - t0:.1f}s", flush=True)
        out.setdefault(rotate, x)
    rel = [float((out[True][:, :, 5 * c:5 * c + 5] - out[False][:, :, 5 * c:5 * c + 5]).norm()
                 / out[False][:, :, 5 * c:5 * c + 5].norm()) for c in range(N_CHUNKS)]
    print("per-chunk rel diff:", " ".join(f"{r:.4f}" for r in rel), flush=True)
    assert max(rel[:6]) < 1e-6, rel                  # no span move yet: identical
    assert max(rel) < TOL, rel


if __name__ == "__main__":
    test_kv_reuse()
    print("ok")
