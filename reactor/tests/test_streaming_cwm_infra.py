"""Small invariants plus optional real-CUDA exactness checks."""

# Optional upstream imports must follow the dependency guards below.

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rtcwm")
pytest.importorskip("musubi_tuner")
from streaming_cwm_infra import install_adaln_bank
from torch import nn


def test_adaln_native_values_and_unknown_schedule():
    from musubi_tuner.minimax_h3.model import AdalnProj, TimeEmbedder
    from rtcwm.model.sampler import schedule

    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.dtype = torch.bfloat16
            self.video_patch_proj = nn.Linear(4, 8)
            self.time_embedder = TimeEmbedder(4, 8, 4)
            self.blocks = nn.ModuleList([AdalnProj(4, 8, 6, 3, dtype=self.dtype)])

    model = Small().eval().requires_grad_(False)
    times = [
        torch.tensor([1 - float(t)], dtype=torch.float32)
        for t in schedule(4, "cpu", 5)[:-1]
    ]
    times += [torch.unique(torch.cat((t, torch.tensor([0.999, 1.0])))) for t in times]
    times += [torch.tensor([0.999, 1.0])]
    with torch.inference_mode():
        expected = [
            model.blocks[0](model.time_embedder(t).to(model.dtype)) for t in times
        ]
        report = install_adaln_bank(model)
        assert report["time_sets"] == 9
        assert not any("linear.weight" in k for k in model.blocks[0].state_dict())
        for t, wanted in zip(times, expected):
            actual = model.blocks[0](model.time_embedder(t).to(model.dtype))
            assert all(torch.equal(a, b) for a, b in zip(actual, wanted))
        with pytest.raises(ValueError, match="Uncached"):
            model.time_embedder(torch.tensor([0.321]))
    with pytest.raises(RuntimeError, match="inference-only"):
        model.time_embedder(times[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_modulation_exact_cuda():
    from musubi_tuner.minimax_h3 import model as h3
    from rtcwm.model import attention
    from streaming_cwm_kernels import install_fused_modulation

    install_fused_modulation()
    generator = torch.Generator(device="cuda").manual_seed(31)
    with torch.inference_mode():
        for rows, width in [(67, 64), (1024, 5376), (5050, 5376)]:
            segments = (
                ((0, rows // 3, 0), (rows // 3, rows // 2, 2), (rows // 2, rows, 1)),
            )
            x = torch.randn(
                (1, rows, width),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            update = torch.randn(
                x.shape, device="cuda", dtype=x.dtype, generator=generator
            )
            # Real AdalnProj returns strided chunks, not standalone small tables.
            table = torch.randn(
                (3, 6 * width), device="cuda", dtype=x.dtype, generator=generator
            )
            shift, scale, gate, *_ = table.chunk(6, -1)
            assert torch.equal(
                h3._mod_scale_shift(x, shift, scale, segments),
                attention.mod_scale_shift(x, shift, scale, segments),
            )
            assert torch.equal(
                h3._mod_gate(x, update, gate, segments),
                attention.mod_gate(x, update, gate, segments),
            )


def test_packed_kv_reuses_only_within_one_chunk():
    from types import MethodType, SimpleNamespace

    from musubi_tuner.minimax_h3.model import Attention
    from rtcwm.model import attention
    from streaming_cwm_infra import PackedKV

    layer = Attention(
        16, 2, 8, 1e-5, attn_mode="torch", split_attn=False, dtype=torch.bfloat16
    ).eval()
    layer._rtcwm_layer = 0
    original = MethodType(attention._forward, layer)
    layer.forward = original
    pack = PackedKV(SimpleNamespace(blocks=[SimpleNamespace(attn=layer)]))
    saved_backend = attention.KV_ATTENTION
    attention.KV_ATTENTION = "sdpa"
    try:
        with torch.inference_mode():
            for chunk in range(2):
                pieces = [
                    [
                        (
                            torch.randn(1, 2, 7, 8, dtype=torch.bfloat16),
                            torch.randn(1, 2, 7, 8, dtype=torch.bfloat16),
                        )
                    ]
                ]
                with pack.chunk():
                    for _ in range(4):
                        hidden = torch.randn(1, 3, 16, dtype=torch.bfloat16)
                        expected_store, actual_store = [], []
                        with attention.active(
                            kv={"past": pieces, "store": expected_store}
                        ):
                            expected = original(hidden)
                        with attention.active(
                            kv={"past": pieces, "store": actual_store}
                        ):
                            actual = layer(hidden)
                        assert torch.equal(actual, expected)
                        assert all(
                            torch.equal(a, b)
                            for a, b in zip(expected_store[0], actual_store[0])
                        )
                    assert len(pack.entries) == 1
                assert not pack.entries and not pack.enabled
    finally:
        attention.KV_ATTENTION = saved_backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_exact_cuda():
    from rtcwm.model import attention
    from streaming_cwm_kernels import install_fused_rope

    original = attention.rope
    install_fused_rope()
    with torch.inference_mode():
        for heads in [7, 14, 56]:
            for dtype in [torch.bfloat16, torch.float16]:
                packed = torch.randn(1, 509, heads, 3 * 128, device="cuda", dtype=dtype)
                x = packed[..., :128]
                for sequence in [1, 509]:
                    rotation = torch.randn(
                        1, sequence, 1, 48, 2, 2, device="cuda", dtype=dtype
                    )
                    assert torch.equal(
                        attention.rope(x, rotation), original(x, rotation)
                    )


@pytest.mark.parametrize("chunks,first", [(0, False), (1, False), (1, True)])
def test_prefix_mask_matches_native_and_hides_future(monkeypatch, chunks, first):
    from rtcwm.model import masks
    from streaming_cwm_prefill import PrefixMask, prefix_attention

    monkeypatch.setattr(masks, "CHUNK_ROWS", 11)
    layout = PrefixMask(5, 18, 3, chunks, first)
    runs, rows = masks.span_runs(5, 18, 3, chunks)
    allowed = torch.zeros(rows, rows, dtype=torch.bool)
    for a, b, role, chunk in runs:
        for c, d, key_role, key_chunk in runs:
            allowed[a:b, c:d] = masks.df_allows(role, chunk, key_role, key_chunk, first)
    generator = torch.Generator().manual_seed(543)
    q, k, v = [
        torch.randn(1, rows, 2, 8, generator=generator, dtype=torch.float64)
        for _ in range(3)
    ]

    def flash(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed
    ).transpose(1, 2)
    actual = prefix_attention(q, k, v, layout, flash)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # Perturb the last proxy: static, earlier proxies and current video must
    # remain exactly unchanged. This is the realtime-control causal boundary.
    changed_k, changed_v = k.clone(), v.clone()
    changed_k[:, 20:23] *= 100
    changed_v[:, 20:23] += 100
    changed = prefix_attention(q, changed_k, changed_v, layout, flash)
    keep = torch.ones(rows, dtype=torch.bool)
    keep[20:23] = False
    assert torch.equal(changed[:, keep], actual[:, keep])


def test_prefix_prefill_stores_native_current_kv(monkeypatch):
    from types import SimpleNamespace

    from musubi_tuner.minimax_h3.model import Attention
    from rtcwm.model import attention, sampler
    from streaming_cwm_prefill import PrefixMask, install_prefix_prefill

    layer = Attention(
        16, 2, 8, 1e-5, attn_mode="torch", split_attn=False, dtype=torch.bfloat16
    ).eval()
    layer._rtcwm_layer = 0
    original = sampler.df_mask
    monkeypatch.setattr(sampler, "df_mask", original)
    monkeypatch.setattr(
        attention,
        "_fa3",
        lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2),
    )
    install_prefix_prefill(SimpleNamespace(blocks=[SimpleNamespace(attn=layer)]))
    mask = sampler.df_mask(5, 18, 3, 0, "cpu", static_sees_first=False)
    assert isinstance(mask, PrefixMask)
    hidden = torch.randn(1, 26, 16, dtype=torch.bfloat16)
    store = []
    with torch.inference_mode():
        _, k, v = layer.qkv_proj(hidden).split(16, -1)
        k = layer.k_norm(k.reshape(1, 26, 2, 8)).transpose(1, 2).contiguous()
        v = v.reshape(1, 26, 2, 8).transpose(1, 2).contiguous()
        with attention.active(mask=mask, kv={"past": None, "store": store}):
            output = layer(hidden)
        assert output.shape == hidden.shape and len(store) == 1
        assert torch.equal(store[0][0], k) and torch.equal(store[0][1], v)
