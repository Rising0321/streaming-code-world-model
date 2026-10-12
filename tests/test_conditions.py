"""Condition re-run of a moved span (GPU): running only the proxy rows against the shared static pieces ==
running all condition rows under the span mask (bf16 / kernel rounding), and the time of each.

    python tests/test_conditions.py
"""
import time

import torch

from rtcwm import paths
from rtcwm.model.loading import load_for_inference
from rtcwm.model.objectives import make_window
from rtcwm.model.sampler import KVSampler
from rtcwm.train.dataset import load_window, silence

TOL = 0.02


def timed(f):
    torch.cuda.synchronize(); t0 = time.time()
    f()
    torch.cuda.synchronize()
    return time.time() - t0


def test_conditions(clip="clip_0244", j=6):
    dev = torch.device("cuda", 0)
    model = load_for_inference(paths.RUNS / "dmd_v4/student_iter00100.safetensors", dev)
    with torch.inference_mode():
        win = make_window(model, load_window(clip, j, silence(), start=0, with_target=False), dev,
                          torch.Generator().manual_seed(0))
        full, part = KVSampler(model, win), KVSampler(model, win)
        full.add_conditions()
        t_full = timed(full.add_conditions)
        static = [(layer[0], layer[-1]) for layer in full.ca_kv]
        part.add_conditions(static)
        t_part = timed(lambda: part.add_conditions(static))
    err = max(max(float((a[i] - b[i]).float().norm() / a[i].float().norm()) for i in (0, 1))
              for la, lb in zip(full.ca_kv, part.ca_kv) for a, b in zip(la[1:-1], lb[1:-1]))
    print(f"proxy pieces, max rel diff over layers / chunks / k,v: {err:.4f}; time {t_full:.2f}s -> {t_part:.2f}s", flush=True)
    assert err < TOL, err


if __name__ == "__main__":
    test_conditions()
    print("ok")
