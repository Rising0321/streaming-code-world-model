"""The DF span mask == the plain role/chunk visibility rule (rtcwm.model.masks.df_allows), on random and
boundary token pairs, for spans that start the stream and spans that do not (needs a GPU).

    python -m pytest tests/test_masks.py      (or: python tests/test_masks.py)
"""
import torch

from rtcwm.model.layout import NUM_CHUNKS
from rtcwm.model.masks import df_allows, df_mask, span_runs

HEAD, DUV_ROWS, AUD = 16300, 1980, 340


def role_chunk(runs):
    starts = torch.tensor([a for a, *_ in runs])

    def rc(i):
        a, b, r, c = runs[int(torch.searchsorted(starts, torch.tensor(i), right=True)) - 1]
        assert a <= i < b
        return r, c
    return rc


def _check(mask, S, rc, edges, ssf):
    g = torch.Generator().manual_seed(0)
    q = torch.randint(0, S, (80000,), generator=g)
    k = torch.randint(0, S, (80000,), generator=g)
    e = torch.tensor(sorted({x + d for x in edges for d in (-1, 0)} | {S - 1}))
    e = e[(e >= 0) & (e < S)]
    q = torch.cat([q, e.repeat_interleave(len(e))])
    k = torch.cat([k, e.repeat(len(e))])
    got = mask.mask_mod(0, 0, q.cuda(), k.cuda()).cpu()
    want = torch.tensor([df_allows(*rc(int(a)), *rc(int(b)), ssf) for a, b in zip(q, k)])
    assert torch.equal(got, want), int((got != want).sum())
    dense = mask.to_dense()[0, 0].cpu().bool()
    assert dense[q[want] // 128, k[want] // 128].all()        # every allowed pair lies in a visited tile


def test_df_masks():
    for ssf in (True, False):
        for n, first in ((NUM_CHUNKS, 0), (1, 0), (3, 2), (1, 4)):
            runs, S = span_runs(HEAD, DUV_ROWS, AUD, n, first)
            _check(df_mask(HEAD, DUV_ROWS, AUD, n, "cuda", first_chunk=first, static_sees_first=ssf), S,
                   role_chunk(runs), [a for a, *_ in runs], ssf)


if __name__ == "__main__":
    test_df_masks()
    print("masks OK")
