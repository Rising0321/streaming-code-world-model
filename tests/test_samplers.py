"""The streaming sampler == the full packed training forward (loads the 33B model on one GPU).

  * rope: the element-wise RoPE equals musubi's batched-matmul one on a real window's rotation table
  * df: the KV-cache step of chunk c, after caching chunks 0..c-1, equals df_forward on the whole window

    python tests/test_samplers.py [CLIP:WINDOW_START_CHUNK]
"""
import sys

import torch

from rtcwm import paths
from rtcwm.model.layout import FRAME_ROWS, NUM_CHUNKS, chunk_latents
from rtcwm.model.loading import load_for_inference
from rtcwm.model.objectives import df_forward, make_window
from rtcwm.model.sampler import KVSampler
from rtcwm.train.dataset import load_window, silence

TOL = 0.03          # vs the 6-chunk training forward: bf16 / flex numerics of a longer sequence, up to ~2.3%
EXACT = 1e-3        # chunk 0 vs the same packing without a cache (ops/dbg/chunk0_numeric.py: 0.0000)


def rel(a, b):
    return float((a.float() - b.float()).norm() / a.float().norm())


def main(clip="clip_0002", j=7, t=0.6):
    dev = torch.device("cuda", 0)
    model = load_for_inference(paths.CWM_LORA, dev)        # outside inference mode: the LoRA merge writes weights
    with torch.inference_mode():
        for jj in (j, 0):                                   # a span inside the stream, and the stream start
            s = load_window(clip, jj, silence())
            if jj == j:
                _check_rope(model, s, dev)
            for c in (0, 3, NUM_CHUNKS - 1):                # the chunk that runs with the conditions, a middle, the last
                _check(model, s, c, t, dev)


def _check_rope(model, s, dev):
    from rtcwm.model.attention import _orig_rope, rope
    win = make_window(model, s, dev, torch.Generator().manual_seed(0))
    pos = torch.cat([win.ca_pos, win.target_pos])
    table = model._rotation_table(pos.unsqueeze(0).to(dev), torch.bfloat16)
    q = torch.randn((1, pos.shape[0], 8, 128), generator=torch.Generator().manual_seed(2)).to(dev, torch.bfloat16)
    r = rel(_orig_rope(q, table), rope(q, table))
    print(f"rope element-wise vs batched matmul: {r:.5f}", flush=True)
    assert r < 1e-3, r


def _check(model, s, c, t, dev):
    x = s["x"].to(dev)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(1)).to(dev)
    sl = chunk_latents(c)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        win = make_window(model, s, dev, torch.Generator().manual_seed(0))
        v = x.clone()
        v[:, :, sl] = t * x[:, :, sl] + (1 - t) * noise[:, :, sl]
        v[:, :, sl.stop:] = noise[:, :, sl.stop:]
        full = df_forward(model, win, v, [1.0] * c + [t] + [1e-4] * (NUM_CHUNKS - c - 1), {}, train=False)[:, :, sl]
        kvs = KVSampler(model, win)
        for i in range(c):
            kvs.add_chunk(i, x[:, :, chunk_latents(i)])
        part = kvs.step(c, v[:, :, sl], t)
        r = rel(full, part)
        if c == 0:
            from rtcwm.model.masks import df_mask
            from rtcwm.model.packing import run_blocks, video_output
            n = sl.stop * FRAME_ROWS
            m1 = df_mask(win.head_rows, win.duv_rows, win.audio_rows, 1, dev, static_sees_first=win.static_sees_first)
            h, temb, inv = run_blocks(model, torch.cat([win.rows, win.video_rows(model, v[:, :, sl])], 1),
                                      torch.cat([win.cond_times(), torch.full((n,), t)]),
                                      torch.cat([win.tags, torch.zeros(n, dtype=torch.int64)]),
                                      torch.cat([win.ca_pos, win.target_pos[:n]]), mask=m1)
            e = rel(video_output(model, h, temb, inv, win.rows.shape[1], sl.stop), part)
            print(f"stream_start={s['stream_start']} chunk 0: KV vs uncached packing: {e:.5f}", flush=True)
            assert e < EXACT, e
    print(f"stream_start={s['stream_start']} chunk {c}: df KV-cache vs full: {r:.4f}", flush=True)
    assert r < TOL, r


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cl, jj = sys.argv[1].split(":")
        main(cl, int(jj))
    else:
        main()
