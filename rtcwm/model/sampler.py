"""Chunk-autoregressive streaming generation.

KVSampler works inside one packed span (rtcwm.model.packing.Window): chunk 0 runs together with the
condition rows (fixed condition times, the span mask of rtcwm.model.masks); a finished chunk runs once more
at model time 1 to add its keys/values (chunk 0 also the condition ones, kept per piece: text + Picture 1,
each proxy chunk, audio), and a denoising step of chunk c >= 1 runs only its own rows against the cache --
the static pieces, proxy chunks <= c (no future control) and the history. Equal to the full packed
forward (tests/test_samplers.py).
Step rules:
    euler : the CWM flow sampler (velocity Euler steps), many steps
    x0    : few-step (distilled students): predict the clean chunk, re-noise it to the next level

generate_stream produces a whole clip chunk by chunk, with no restart. Stream chunk k is generated with
its attention span, chunks j..k (j = span_start(k)), packed with that span's conditions and positions;
the history chunks j..k-1 (already generated, clean) are re-run into the cache first. This is exactly the
training situation of diffusion forcing with a clean prefix. kv_reuse=True instead keeps the history's keys
and values across chunks (moved to the new positions), re-running only the conditions -- faster, approximate.
"""
from __future__ import annotations

from typing import Callable

import torch

from cwm_h3_inference.constants import VIDEO_SHIFT

from .layout import CLEAN_T, NUM_CHUNKS, SPAN_CHUNKS, chunk_row_slice, chunk_shape, span_start
from .masks import df_mask
from .objectives import make_window, shift
from .packing import Window, run_blocks, video_output

def schedule(steps: int, device, shift_value: float = VIDEO_SHIFT) -> torch.Tensor:
    """steps + 1 noise levels from 1 to 0, flow-shifted."""
    return shift(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64, device=device), shift_value).float()


class KVSampler:
    """`train=True` builds the cache under no_grad (its tensors may then enter an autograd graph) and
    checkpoints gradient steps; otherwise everything runs under inference_mode."""

    def __init__(self, model, win: Window, train: bool = False):
        self.model, self.win, self.train = model, win, train
        self.ca_kv = None                      # per layer: condition pieces [head, proxy 0..5, audio], cached with chunk 0
        self.chunks_kv = []                    # per finished chunk: [(k, v)] per layer; None once out of the window

    def _no_grad(self):
        return torch.no_grad() if self.train else torch.inference_mode()

    def past(self, c: int):
        if c == 0:
            return None                        # chunk 0 runs with the condition rows themselves
        hist = self.chunks_kv[max(0, c - (SPAN_CHUNKS - 1)):c]
        assert all(h is not None for h in hist)
        return [[self.ca_kv[li][0]] + self.ca_kv[li][1:2 + c] + [self.ca_kv[li][-1]] + [h[li] for h in hist]
                for li in range(len(self.ca_kv))]

    def step(self, c: int, latents, model_t: float, past=None, store=None, grad: bool = False):
        """One forward of chunk c (latents [1,24,5,H,W]) at model time model_t -> its velocity."""
        if grad:
            # autocast caches the bf16 casts of the (fp32) LoRA weights for the whole autocast region; the
            # casts made by earlier no-grad steps carry no autograd history and would cut the gradient
            torch.clear_autocast_cache()
        ctx = torch.enable_grad() if grad else self._no_grad()
        with ctx:
            win = self.win
            hv = win.video_rows(self.model, latents)
            t_rows = torch.full((hv.shape[1],), float(model_t))
            tags = torch.zeros(hv.shape[1], dtype=torch.int64)
            pos = win.target_pos[chunk_row_slice(c)]
            if c == 0:
                if getattr(self, "mask0", None) is None:
                    self.mask0 = df_mask(win.head_rows, win.duv_rows, win.audio_rows, 1, win.dev,
                                         static_sees_first=win.static_sees_first)
                hidden, temb, inv = run_blocks(self.model, torch.cat([win.rows, hv], 1),
                                               torch.cat([win.cond_times(), t_rows]), torch.cat([win.tags, tags]),
                                               torch.cat([win.ca_pos, pos]), mask=self.mask0, train=grad,
                                               kv=dict(past=None, store=store))
                return video_output(self.model, hidden, temb, inv, win.rows.shape[1], latents.shape[2])
            hidden, temb, inv = run_blocks(self.model, hv, t_rows, tags, pos, train=grad,
                                           kv=dict(past=past if past is not None else self.past(c), store=store))
            return video_output(self.model, hidden, temb, inv, 0, latents.shape[2])

    def add_conditions(self, static=None):
        """Cache the condition pieces alone: a span that does not start the stream, whose condition rows see
        no video row (rtcwm.model.masks), so they come out exactly as in add_chunk(0, ...). static: the
        static pieces [(head, audio)] per layer of an earlier such span -- the same in every one -- so only
        the proxy rows are run, each proxy chunk seeing the static rows and the proxy chunks up to it."""
        assert not self.win.static_sees_first
        win, store = self.win, []
        if static is None:
            mask = df_mask(win.head_rows, win.duv_rows, win.audio_rows, 0, win.dev, static_sees_first=False)
            with self._no_grad():
                run_blocks(self.model, win.rows, win.cond_times(), win.tags, win.ca_pos, mask=mask,
                           kv=dict(past=None, store=store))
            self.ca_kv = [self._pieces(k, v) for k, v in store]
            return
        rows = slice(win.head_rows, win.head_rows + win.duv_rows)
        dr = win.duv_rows // NUM_CHUNKS
        with self._no_grad():
            run_blocks(self.model, win.rows[:, rows], win.cond_times()[rows], win.tags[rows], win.ca_pos[rows],
                       kv=dict(past=[[head, audio] for head, audio in static], store=store, causal_blocks=dr))
        self.ca_kv = [[head] + [(k[:, :, a:a + dr].clone(), v[:, :, a:a + dr].clone()) for a in range(0, win.duv_rows, dr)]
                      + [audio] for (head, audio), (k, v) in zip(static, store)]

    def _pieces(self, k, v):
        """Condition keys/values [B,H,rows,D] -> pieces [head, proxy 0..5, audio] (copies: views would keep
        the whole run alive)."""
        w = self.win
        dr = w.duv_rows // NUM_CHUNKS
        cuts = [0, w.head_rows] + [w.head_rows + (p + 1) * dr for p in range(NUM_CHUNKS)] + [w.rows.shape[1]]
        return [(k[:, :, a:b].clone(), v[:, :, a:b].clone()) for a, b in zip(cuts[:-1], cuts[1:])]

    def add_chunk(self, c: int, latents, past=None, model_t: float = CLEAN_T):
        """Cache the keys/values of finished chunk c (run at model time model_t: 1 = clean)."""
        store = []
        self.step(c, latents.detach(), model_t, past=past, store=store)
        assert len(store) == len(self.model.blocks)
        if c == 0:                                            # split off the condition pieces
            n = self.win.rows.shape[1]
            self.ca_kv = []
            for li in range(len(store)):                      # layer by layer
                k, v = store[li]
                self.ca_kv.append(self._pieces(k[:, :, :n], v[:, :, :n]))
                store[li] = (k[:, :, n:].clone(), v[:, :, n:].clone())
                del k, v
        self.chunks_kv.append(store)
        old = c + 1 - SPAN_CHUNKS                           # no later chunk looks at it any more
        if old >= 0:
            self.chunks_kv[old] = None

    def generate_chunk(self, c: int, noise, sigmas, rule: str = "euler", gen=None, grad_step: int | None = None,
                       cache: bool = True, exit_step: int | None = None, keep_last_kv: bool = False):
        """Denoise chunk c from `noise` over `sigmas`. x0 rule only: with exit_step=k the chunk is the clean
        prediction of step k (self forcing's random exit; later steps are skipped), with grad_step=k also
        with its graph kept. cache: add the finished chunk to the cache (not needed when no later chunk of
        this window follows). keep_last_kv: keep the keys/values the chunk's rows had in its last step (the
        one its output comes from) in self.last_kv -- a history without the extra clean forward of
        add_chunk (HISTORY_FROM_LAST_STEP)."""
        past = self.past(c)
        x, out = noise, None
        stop = grad_step if grad_step is not None else exit_step
        for i in range(len(sigmas) - 1):
            s, s_next = float(sigmas[i]), float(sigmas[i + 1])
            g = grad_step == i
            last = i == stop or i == len(sigmas) - 2
            store = [] if keep_last_kv and last else None
            v = self.step(c, x, 1.0 - s, past=past, grad=g, store=store).float()
            if store is not None:
                self.last_kv = [(k.detach(), v_.detach()) for k, v_ in store]
            if rule == "euler":
                x = x + (s - s_next) * v
                continue
            x0 = x + s * v                                     # noisy = (1-s) x0 + s n, v = x0 - n
            if i == stop or i == len(sigmas) - 2:
                out = x0
                break
            with self._no_grad():
                x = (1 - s_next) * x0 + s_next * torch.randn(x0.shape, generator=gen).to(x0.device)
        out = x if rule == "euler" else out
        if cache and c < NUM_CHUNKS - 1:
            self.add_chunk(c, out, past=past)
        return out


ROTATE_IN_ATTENTION = True   # kv_reuse: cached keys keep their positions, moved by rotation inside attention
HISTORY_FROM_LAST_STEP = True   # kv_reuse, sampling only: cache a chunk with the keys/values of its last denoising
                                # step (noisy input) instead of one more clean forward -- one forward fewer per chunk,
                                # and the noisy history damps the drift of a long rollout (a training rollout gets the
                                # same history from rollout(history_noise=...))
PROXY_ONLY_CONDITIONS = False   # kv_reuse: on a span move re-run only the proxy rows against the shared static
                                # pieces (experimental, pending tests/test_conditions.py) instead of all conditions


def _move(model, kv, delta: float, dev):
    """[cached chunk, its pending time shift] after the span moved by delta. Normally the keys stay as they
    were cached and only the shift accumulates (_moved hands it to attention), so every user of a chunk --
    including the autograd graphs of a training rollout, which keep what they attended to until the
    backward pass -- shares one copy. ROTATE_IN_ATTENTION = False re-rotates a copy of the keys at every
    move (the earlier behaviour, kept for tests)."""
    store, shift = kv
    if not ROTATE_IN_ATTENTION:
        return [_shift_time(model, store, delta, dev), 0.0]
    return [store, shift + delta]


def _moved(model, store, shift: float, dev, tables: dict):
    """Per-layer pieces of a cached chunk at its current position: (k, v) or (k, v, rotation by shift)."""
    if shift == 0:
        return store
    tab = tables.get(shift)
    if tab is None:
        tab = tables[shift] = model._rotation_table(torch.tensor([[[shift, 0.0, 0.0]]], device=dev), store[0][0].dtype)
    return [(k, v, tab) for k, v in store]


def _share_static(ca, static, sees_first: bool):
    """Static condition pieces (caption + Picture 1, audio) of spans that do not start the stream see only
    static rows at fixed positions, so they are the same in every such span: keep the first span's and
    hand it to the later ones (the re-run's own copies are dropped), one copy instead of one per span."""
    if sees_first:
        return static
    if static is None:
        return [(layer[0], layer[-1]) for layer in ca]
    for layer, (head, audio) in zip(ca, static):
        layer[0], layer[-1] = head, audio
    return static


def _shift_time(model, store, delta: float, dev):
    """Cached keys re-positioned by delta on the time axis: RoPE rotations compose, so rotating the keys
    of position p by delta gives exactly the keys of position p + delta (values carry no position)."""
    from .attention import rope
    table = model._rotation_table(torch.tensor([[[delta, 0.0, 0.0]]], device=dev), store[0][0].dtype)
    return [(rope(k.transpose(1, 2), table).transpose(1, 2).contiguous(), v) for k, v in store]


@torch.inference_mode()
def generate_stream(model, window_sample: Callable[[int], dict], n_chunks: int, steps: int, gen, dev,
                    rule: str = "euler", shift_value: float = VIDEO_SHIFT, history_noise: float = 0.0,
                    kv_reuse: bool = False, on_chunk=None) -> torch.Tensor:
    """Stream chunks 0..n_chunks-1 -> [1,24,5*n_chunks,H,W]. window_sample(j) returns the inputs of the
    span starting at stream chunk j (rtcwm.train.dataset.load_window); on_chunk(k, seconds) reports progress.
    history_noise s > 0: the history chunks enter the cache noised to level s (model time 1 - s), as
    diffusion forcing trains them -- damps the drift of a long rollout on its own output.
    kv_reuse: keep each history chunk's keys/values from when it was cached and only move them to the
    new span's positions (a rotation applied inside attention, _move); only the condition rows (whose proxy
    changes with the span) are re-run, alone. Saves the re-run of the history, but its keys and values were
    computed in the earlier spans -- an approximation of the exact default."""
    out = _stream(model, window_sample, n_chunks, schedule(steps, dev, shift_value), gen, dev, rule, history_noise,
                  kv_reuse, on_chunk=on_chunk)
    return torch.cat(out, 2)


def rollout(model, window_sample: Callable[[int], dict], n_chunks: int, sigmas, gen, dev, exit_steps,
            grad_chunks=(), kv_reuse: bool = True, history_noise: float = 0.0, on_chunk=None) -> list[torch.Tensor]:
    """Self-forcing rollout of a training student (x0 rule), the generate_stream procedure except that
    chunk k stops at its step exit_steps[k] and is that step's clean prediction (and so the history of the
    later chunks); the chunks in grad_chunks keep the graph of that step (the history they attend to enters
    detached). Returns the chunks [1,24,5,H,W]. Every rank must pass the same exit_steps / grad_chunks:
    they fix the sequence of block calls, and so of the FSDP gathers. history_noise s > 0: every history
    chunk but the stream's first enters the cache noised to level s -- with s = sigmas[-2] the history a
    HISTORY_FROM_LAST_STEP sampler caches (the last step's input), whatever step the chunk exited at."""
    return _stream(model, window_sample, n_chunks, sigmas, gen, dev, "x0", history_noise, kv_reuse, on_chunk=on_chunk,
                   train=True, exit_steps=exit_steps, grad_chunks=set(grad_chunks), clean_first=True)


def _stream(model, window_sample, n_chunks, sig, gen, dev, rule, history_noise, kv_reuse, on_chunk=None,
            train: bool = False, exit_steps=None, grad_chunks=frozenset(), clean_first: bool = False) -> list[torch.Tensor]:
    import time
    from musubi_tuner.minimax_h3.packing import _video_t_spans
    chunk_span = sum(_video_t_spans(chunk_shape()[2]))       # time-axis length of one chunk
    out, hist_lat, hist_kv, ca, prev_j = [], [], {}, None, None
    static, tables = None, {}                                # see _share_static / _moved
    for k in range(n_chunks):
        t0 = time.time()
        j = span_start(k)
        with torch.no_grad():
            win = make_window(model, window_sample(j), dev, gen)
        kvs = KVSampler(model, win, train=train)
        with kvs._no_grad():
            if kv_reuse:
                if prev_j is not None and j != prev_j:           # the span moved: drop its old first chunk,
                    delta = -(j - prev_j) * chunk_span            # move the rest, re-run the conditions
                    hist_kv = {i: _move(model, kv, delta, dev) for i, kv in hist_kv.items() if i >= j}
                    ca = None
                if k > j and ca is None:                         # (a moved span never starts the stream:
                    kvs.add_conditions(static if PROXY_ONLY_CONDITIONS else None)  # its conditions see no video, and the
                    ca = kvs.ca_kv                                # span's first chunk keeps its keys/values)
                    static = _share_static(ca, static, win.static_sees_first)
                kvs.ca_kv, kvs.chunks_kv = ca, [_moved(model, *hist_kv[i], dev, tables) for i in range(j, k)]
            else:
                for c in range(k - j):                             # the history of this span, re-run
                    kvs.add_chunk(c, *hist_lat[j + c])
        noise = torch.randn(chunk_shape(), generator=gen).to(dev)
        e = exit_steps[k] if exit_steps is not None else None
        last_kv = HISTORY_FROM_LAST_STEP and not train and kv_reuse and k > j and k < n_chunks - 1
        x = kvs.generate_chunk(k - j, noise, sig, rule, gen, grad_step=e if k in grad_chunks else None, cache=False,
                               exit_step=e, keep_last_kv=last_kv).float()
        out.append(x)
        with kvs._no_grad():
            h, hn = x.detach(), 0.0 if clean_first and k == 0 else history_noise
            if hn > 0:
                h = (1 - hn) * h + hn * torch.randn(h.shape, generator=gen).to(dev)
            hist_lat.append((h, 1.0 - hn))
            if kv_reuse and k < n_chunks - 1:                      # cache the new chunk for the next spans
                if k == j:                                        # (only k = 0: it runs with the conditions)
                    kvs.add_chunk(0, h, model_t=1.0 - hn)
                    ca, hist_kv[k] = kvs.ca_kv, [kvs.chunks_kv[0], 0.0]
                    static = _share_static(ca, static, win.static_sees_first)
                elif last_kv:
                    hist_kv[k] = [kvs.last_kv, 0.0]
                else:
                    kvs.add_chunk(k - j, h, model_t=1.0 - hn)
                    hist_kv[k] = [kvs.chunks_kv[-1], 0.0]
        prev_j = j
        del kvs, win
        if on_chunk:
            on_chunk(k, time.time() - t0)
    return out
