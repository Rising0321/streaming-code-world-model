"""Gather/modulation fusion preserving every native BF16 rounding boundary."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scale_shift(
    X,
    SHIFT,
    SCALE,
    ROWS,
    OUT,
    N: tl.constexpr,
    C: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    row = tl.load(ROWS + i // C, valid, 0)
    shift = tl.load(SHIFT + row * TABLE_STRIDE + i % C, valid, 0).to(tl.float32)
    scale = tl.load(SCALE + row * TABLE_STRIDE + i % C, valid, 0).to(tl.float32)
    x = tl.load(X + i, valid, 0).to(tl.float32)
    # Eager BF16 stores a rounded tensor after EACH operator. No FMA.
    factor = (1.0 + scale).to(tl.bfloat16).to(tl.float32)
    product = (x * factor).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, (product + shift).to(tl.bfloat16), valid)


@triton.jit
def _gate(
    RES,
    UPDATE,
    GATE,
    ROWS,
    OUT,
    N: tl.constexpr,
    C: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    row = tl.load(ROWS + i // C, valid, 0)
    gate = tl.load(GATE + row * TABLE_STRIDE + i % C, valid, 0).to(tl.float32)
    update = tl.load(UPDATE + i, valid, 0).to(tl.float32)
    residual = tl.load(RES + i, valid, 0).to(tl.float32)
    product = (update * gate).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, (residual + product).to(tl.bfloat16), valid)


@triton.jit
def _rope(
    X,
    R,
    OUT,
    N: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    XS2: tl.constexpr,
    XS3: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    RS3: tl.constexpr,
    RS4: tl.constexpr,
    RS5: tl.constexpr,
    RB: tl.constexpr,
    RT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    d = i % D
    h = i // D % H
    s = i // (D * H) % S
    b = i // (D * H * S)
    base = b * XS0 + s * XS1 + h * XS2
    pair = d % P
    is_rotary = valid & (d < 2 * P)
    a = tl.load(X + base + pair * XS3, is_rotary, 0).to(tl.float32)
    c = tl.load(X + base + (pair + P) * XS3, is_rotary, 0).to(tl.float32)
    rbase = (b % RB) * RS0 + (s % RT) * RS1 + pair * RS3 + tl.where(d < P, 0, 1) * RS4
    r0 = tl.load(R + rbase, is_rotary, 0).to(tl.float32)
    r1 = tl.load(R + rbase + RS5, is_rotary, 0).to(tl.float32)
    rotated = r0 * a + r1 * c
    original = tl.load(X + base + d * XS3, valid, 0).to(tl.float32)
    tl.store(OUT + i, tl.where(d < 2 * P, rotated, original), valid)


def install_fused_rope():
    from musubi_tuner.minimax_h3 import model as h3
    from rtcwm.model import attention

    original = attention.rope

    def rope(x, table):
        if (
            torch.is_grad_enabled()
            or not x.is_cuda
            or x.dtype not in (torch.bfloat16, torch.float16)
        ):
            return original(x, table)
        if table.ndim != 6 or table.shape[2] != 1:
            raise ValueError("Expected native split-half rotary table")
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        _rope[(triton.cdiv(x.numel(), 1024),)](
            x,
            table,
            out,
            x.numel(),
            x.shape[1],
            x.shape[2],
            x.shape[3],
            table.shape[-3],
            *x.stride(),
            table.stride(0),
            table.stride(1),
            table.stride(3),
            table.stride(4),
            table.stride(5),
            table.shape[0],
            table.shape[1],
            1024,
            enable_fp_fusion=False,
        )
        return out

    attention.rope = h3._apply_rope_split_half = rope


def install_fused_modulation():
    from musubi_tuner.minimax_h3 import model as h3
    from rtcwm.model import attention

    def scale_shift(x, shift, scale, segments):
        if (
            torch.is_grad_enabled()
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.shape[0] != 1
            or not x.is_contiguous()
        ):
            return attention.mod_scale_shift(x, shift, scale, segments)
        idx = attention._rows(segments, x.device)
        out = torch.empty_like(x)
        _scale_shift[(triton.cdiv(x.numel(), 1024),)](
            x,
            shift,
            scale,
            idx,
            out,
            x.numel(),
            x.shape[-1],
            shift.stride(0),
            1024,
            enable_fp_fusion=False,
        )
        return out

    def gate(residual, update, values, segments):
        if (
            torch.is_grad_enabled()
            or residual.dtype != torch.bfloat16
            or not residual.is_cuda
            or residual.shape[0] != 1
            or not residual.is_contiguous()
            or not update.is_contiguous()
        ):
            return attention.mod_gate(residual, update, values, segments)
        idx = attention._rows(segments, residual.device)
        out = torch.empty_like(residual)
        _gate[(triton.cdiv(residual.numel(), 1024),)](
            residual,
            update,
            values,
            idx,
            out,
            residual.numel(),
            residual.shape[-1],
            values.stride(0),
            1024,
            enable_fp_fusion=False,
        )
        return out

    h3._mod_scale_shift, h3._mod_gate = scale_shift, gate
