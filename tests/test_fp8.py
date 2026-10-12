"""FP8 linears (GPU, Hopper): FP8Linear vs the bf16 linear at the sampler's shapes (one chunk's 5,040 rows
through each block linear) -- relative error and time.

    python tests/test_fp8.py
"""
import time

import torch
from torch import nn

from rtcwm.model.fp8 import FP8Linear

TOL = 0.05                                    # FP8 rounding (e4m3: 3 mantissa bits)


def bench(f, x, n=20):
    for _ in range(3):
        f(x)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        f(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e3


def test_fp8(rows=5040):
    torch.manual_seed(0)
    for name, (i, o) in dict(qkv=(5376, 16128), out=(5376, 5376), fc1=(5376, 28672), fc2=(14336, 5376)).items():
        lin = nn.Linear(i, o, bias=False, device="cuda", dtype=torch.bfloat16)
        q = FP8Linear(lin)
        x = torch.randn(1, rows, i, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            ref, got = lin(x).float(), q(x).float()
            e = float((got - ref).norm() / ref.norm())
            tb, tq = bench(lin, x), bench(q, x)
        print(f"{name:4s} {i}x{o}: rel err {e:.4f}  bf16 {tb:.2f} ms -> fp8 {tq:.2f} ms", flush=True)
        assert e < TOL, (name, e)


if __name__ == "__main__":
    test_fp8()
    print("ok")
