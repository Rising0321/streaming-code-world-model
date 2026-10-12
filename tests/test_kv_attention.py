"""KV-cache attention (GPU): FlashAttention-3 over the concatenated past == torch SDPA (the earlier
behaviour), at the sampler's sizes. (An autoregressive stream amplifies bf16 rounding chunk after chunk, so
whole streams are compared by evaluation metrics, not here.)

    python tests/test_kv_attention.py
"""

import torch

from rtcwm.model import attention

TOL_OP = 0.01


def test_ops():
    g = torch.Generator("cuda").manual_seed(0)
    r = lambda n: torch.randn(1, 48, n, 128, device="cuda", generator=g, dtype=torch.bfloat16)
    q = r(5040)
    pieces = [(r(n), r(n)) for n in (3655, 330, 330, 340, 5040, 5040, 5040, 5040, 5040, 5040)]
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, torch.cat([k for k, _ in pieces], 2), torch.cat([v for _, v in pieces], 2)).transpose(1, 2).float()
    cat = attention._fa3(q.transpose(1, 2), torch.cat([k for k, _ in pieces], 2).transpose(1, 2),
                         torch.cat([v for _, v in pieces], 2).transpose(1, 2)).float()
    e = float((cat - ref).norm() / ref.norm())
    print(f"op: fa3 vs sdpa {e:.5f}", flush=True)
    assert e < TOL_OP


if __name__ == "__main__":
    test_ops()
    print("ok")
