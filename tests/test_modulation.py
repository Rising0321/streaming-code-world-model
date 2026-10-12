"""Vectorised AdaLN modulation == musubi's per-segment loop (CPU): outputs and the gradients that reach the
hidden states (and through them the LoRA) bit for bit; the gradients of shift / scale / gate -- they only
reach the frozen AdaLN projection -- up to bf16 accumulation order.

    python tests/test_modulation.py
"""
import torch

from rtcwm.model.attention import _orig_mod_gate, _orig_mod_scale_shift, mod_gate, mod_scale_shift


def plan(S, rows, g):
    cuts = sorted(set(torch.randint(1, S, (300,), generator=g).tolist()))
    bounds = [0] + cuts + [S]
    return (tuple((a, b, int(torch.randint(0, rows, (1,), generator=g))) for a, b in zip(bounds[:-1], bounds[1:])),)


def test_modulation():
    g = torch.Generator().manual_seed(0)
    S, D, R = 4000, 64, 9
    segs = plan(S, R, g)
    x0 = torch.randn(1, S, D, generator=g).to(torch.bfloat16)
    u0 = torch.randn(1, S, D, generator=g).to(torch.bfloat16)
    mods = [torch.randn(R, D, generator=g).to(torch.bfloat16) for _ in range(3)]
    outs = []
    for ss, gt in ((_orig_mod_scale_shift, _orig_mod_gate), (mod_scale_shift, mod_gate)):
        x, upd = x0.clone().requires_grad_(), u0.clone().requires_grad_()
        shift, scale, gate = (m.clone().requires_grad_() for m in mods)
        y = gt(x, ss(x * 1, shift, scale, segs) * upd, gate, segs)
        y.float().square().sum().backward()
        outs.append(dict(y=y.detach(), dx=x.grad, dupd=upd.grad, dshift=shift.grad, dscale=scale.grad, dgate=gate.grad))
    a, b = outs
    for k in ("y", "dx", "dupd"):
        assert torch.equal(a[k], b[k]), k
    for k in ("dshift", "dscale", "dgate"):
        rel = float((a[k].float() - b[k].float()).norm() / a[k].float().norm())
        assert rel < 0.1, (k, rel)                   # bf16 sums over thousands of rows, different order
    print("modulation OK: output and hidden-state gradients bit-exact")


if __name__ == "__main__":
    test_modulation()
