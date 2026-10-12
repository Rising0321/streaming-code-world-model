"""FlashAttention-3 prefix calls (attention.dense_df) == flex attention under df_mask (GPU): outputs and
q / k / v gradients of a real-size training span, with and without the stream start; then the time of
both per layer.

    python tests/test_dense_attention.py
"""
import time

import torch

from rtcwm.model.attention import _compiled_flex, dense_df
from rtcwm.model.layout import NUM_CHUNKS
from rtcwm.model.masks import DenseDF, df_mask

HEAD, DUV, AUDIO = 2647 + 1008, 1980, 340            # caption + native Picture 1, proxy, audio rows
TOL = 0.02                                             # bf16 accumulation order: relative max error


def run(fn, q, k, v, g):
    q, k, v = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o = fn(q, k, v)
    o.backward(g)
    return o.detach().float(), [t.grad.float() for t in (q, k, v)]


def flex(lay, dev):
    m = df_mask(lay.head_rows, lay.duv_rows, lay.audio_rows, lay.n_chunks, dev, static_sees_first=lay.static_sees_first)
    return lambda q, k, v: _compiled_flex(*(t.transpose(1, 2) for t in (q, k, v)), block_mask=m).transpose(1, 2)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def test_dense_df(heads=4, bench=False):
    dev = torch.device("cuda")
    g = torch.Generator(dev).manual_seed(0)
    for ssf in (True, False):
        lay = DenseDF(HEAD, DUV, AUDIO, NUM_CHUNKS, ssf)
        S = HEAD + DUV + AUDIO + NUM_CHUNKS * 5040
        q, k, v, dout = (torch.randn(1, S, heads, 128, device=dev, generator=g, dtype=torch.bfloat16) for _ in range(4))
        ref, ref_g = run(flex(lay, dev), q, k, v, dout)
        out, out_g = run(lambda *t: dense_df(*t, lay), q, k, v, dout)
        errs = [rel(out, ref)] + [rel(a, b) for a, b in zip(out_g, ref_g)]
        print(f"static_sees_first={ssf} S={S}: rel max error out / dq / dk / dv " + " ".join(f"{e:.4f}" for e in errs))
        assert max(errs) < TOL, errs
        if bench:
            for name, fn in (("flex", flex(lay, dev)), ("fa3", lambda *t: dense_df(*t, lay))):
                qq, kk, vv = (t.detach().clone().requires_grad_() for t in (q, k, v))
                for i in range(6):
                    if i == 3:
                        torch.cuda.synchronize(); t0 = time.time()
                    o = fn(qq, kk, vv)
                    if i == 3:
                        torch.cuda.synchronize(); t1 = time.time()
                    o.backward(dout)
                    if i == 3:
                        torch.cuda.synchronize(); t2 = time.time()
                print(f"  {heads} heads {name}: forward {1e3 * (t1 - t0):.0f} ms  backward {1e3 * (t2 - t1):.0f} ms")


if __name__ == "__main__":
    test_dense_df()
    test_dense_df(heads=48, bench=True)
    print("ok")
