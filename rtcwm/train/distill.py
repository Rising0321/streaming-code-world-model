"""Few-step distillation of the streaming causal CWM by distribution matching (DMD) on the student's own
long rollouts (self forcing), as in CausVid / Self Forcing / Self Forcing++.

    torchrun ... -m rtcwm.train.distill --out DIR --student-lora DF_LORA [--steps 4] [--iters 200]

Three LoRAs on one frozen, FSDP-sharded H3 (rtcwm.model.loading.LoRAStack):
  student  the streaming DF model (rtcwm.train.train), sampled chunk by chunk with `--steps` x0-steps
  teacher  the released CWM LoRA, frozen: the real score
  critic   initialised from the released CWM LoRA, trained to denoise the student's samples: the fake score
The teacher and the critic run the released model exactly as released: bidirectional over a released
37-latent window (124 frames = 7 VAE clips + the first 2 latents of the 8th, a 5n+2 window), with that
window's whole proxy. The VAE is causal inside a clip, so 37 consecutive latents of a chunk stream are
exactly the released encoding of those frames (ops/dbg/tail_latents.py) -- no layout mismatch.

Each iteration (rollout length R shared by all ranks, drawn from --min-chunks..--max-chunks):
  1. every rank draws a clip and a start frame m and streams R chunks from it (rtcwm.model.sampler.rollout,
     the inference procedure, with the KV cache carried across chunks). As in Self Forcing (mingjia's
     port, wt-sf-mingjia/sf/self_forcing_model.py), every chunk stops at its own random x0-step (shared by
     all ranks) and that step's clean prediction is the chunk -- and the history of the next ones. The
     last --grad-chunks chunks of the teacher window keep the graph of their stopping step, their history
     entering detached (memory: a graph keeps what it attended to -- its span's proxy keys/values and the
     history chunks that have since left the window -- alive until the backward pass; four fit).
     Earlier chunks keep no graph: the window slides, and a graph would keep its own shifted copy of the
     history's keys/values alive until the backward pass.
     A history chunk enters the cache as the sampler caches it (rtcwm.model.sampler.HISTORY_FROM_LAST_STEP):
     noised to the last sampling step's level, the first chunk clean (--rollout-history last-step), or clean.
  2. teacher window = chunks R-8 .. R-2 and the first 2 latents of chunk R-1 (Picture 1 = the window's own
     first frame from the rollout, as the released CWM is used; the start frame's caption; the window's
     proxy; silent audio), noised to one random level:
        loss = 0.5 || x - sg(x - (x0_fake - x0_real) / mean|x - x0_real|) ||^2  over the window  (DMD gradient)
  3. critic: --critic-per-gen flow-matching updates (generator : critic = 1 : 7 by default), each on a
     random teacher window of the same detached rollout (R-7 of them) with fresh noise -- the rollout, the
     expensive part, is shared.
With --teacher-kind causal the teacher is a streaming DF checkpoint (--teacher-lora, frozen) and the critic
starts from it: the teacher window is then the student's own layout -- one 6-chunk span of the rollout
(the last one for the generator, a random one for the critic) with the stream's start frame and caption and
the span's causal proxy (rule E, rtcwm.model.masks) -- every chunk of it noised to the same level.
Long rollouts put the student in the drifted states it reaches at test time, and the teacher's score pulls
them back -- training on its own history removes the exposure bias of diffusion forcing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from cwm_h3_inference.constants import VIDEO_LATENT_SHAPE, VIDEO_SHIFT

from .. import paths
from ..model import loading
from ..model.layout import CHUNK_LATENTS, SPAN_CHUNKS
from ..model.objectives import df_forward, make_window, shift
from ..model.packing import Window, run_blocks, video_output
from ..model.sampler import rollout, schedule
from .dataset import list_samples, load_window, silence, start_frames, stream_chunks
from .train import node_group

STUDENT_SHIFT = 5.0                    # few-step schedule: sigma 1, .94, .83, .63, 0 for 4 steps
TEACHER_LATENTS = VIDEO_LATENT_SHAPE[2]                       # 37 = 7 chunks + 2
TEACHER_CHUNKS = TEACHER_LATENTS // CHUNK_LATENTS + 1         # chunks a teacher window touches: 8


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--student-lora", type=Path, required=True, help="streaming DF checkpoint to distil")
    ap.add_argument("--teacher-lora", type=Path, default=paths.CWM_LORA)
    ap.add_argument("--critic-lora", type=Path, default=paths.CWM_LORA)
    ap.add_argument("--steps", type=int, default=4, help="student sampling steps")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--min-chunks", type=int, default=TEACHER_CHUNKS, help="shortest rollout (chunks)")
    ap.add_argument("--max-chunks", type=int, default=14, help="longest rollout (chunks; 14 = 10 s)")
    ap.add_argument("--kv-reuse", choices=["on", "off"], default="on", help="rollout: carry the history's keys/values")
    ap.add_argument("--rollout-history", choices=["last-step", "clean"], default="last-step",
                    help="rollout history: last-step = noised to the last sampling step's level, the history the sampler "
                         "caches by default (HISTORY_FROM_LAST_STEP); clean = one more clean forward")
    ap.add_argument("--gen-lr", type=float, default=2e-6)
    ap.add_argument("--critic-lr", type=float, default=4e-6)
    ap.add_argument("--critic-per-gen", type=int, default=7, help="critic updates per generator update")
    ap.add_argument("--grad-chunks", type=int, default=4, help="last teacher-window chunks that keep their graph")
    ap.add_argument("--teacher-kind", choices=["released", "causal"], default="released",
                    help="released: the released CWM on 37-latent bidirectional windows; causal: a streaming DF "
                         "checkpoint (--teacher-lora, also the critic's start) on the student's 6-chunk spans")
    ap.add_argument("--teacher-picture1", choices=["window", "start"], default="window",
                    help="teacher / critic Picture 1: the window's first frame (rollout) or the stream's start frame")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--val-clips", type=int, default=12)
    ap.add_argument("--picture1", choices=["native", "2048"], default="native")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def park(opts, device):
    """Move the optimizers' moments to `device`: the CPU while the rollout's graphs fill the GPU (about
    10 GB for student + critic), back before a step."""
    for opt in opts:
        for state in opt.state.values():
            for key, val in state.items():
                if torch.is_tensor(val) and val.dim() > 0:
                    state[key] = val.to(device, non_blocking=True)


class Reducer:
    """Asynchronous gradient all-reduce for replicated LoRA weights, overlapped with backward."""

    def __init__(self, params):
        self.handles = []
        for p in params:
            p.register_post_accumulate_grad_hook(
                lambda q: self.handles.append(dist.all_reduce(q.grad, op=dist.ReduceOp.AVG, async_op=True)))

    def wait(self):
        for h in self.handles:
            h.wait()
        self.handles.clear()


class Starts:
    """(clip, start frame m) pairs whose rollout of n chunks stays inside the training spans (no held-out
    clip, no QC-flagged span) and whose teacher window fits the stream."""

    def __init__(self, samples, res: str):
        spans = {}
        for c, j in samples:
            spans.setdefault(c, set()).add(j)
        self.spans, self.frames = spans, {c: start_frames(c, res) for c in spans}
        self.n_stream = {c: stream_chunks(c) for c in spans}

    def draw(self, rng: random.Random, n: int):
        while True:
            clip = rng.choice(sorted(self.spans))
            ok = [m for m in self.frames[clip] if m + n <= self.n_stream[clip]
                  and all(m + j in self.spans[clip] for j in range(0, n - SPAN_CHUNKS + 1))]
            if ok:
                return clip, rng.choice(ok)


def released_velocity(model, win: Window, latents, model_t: float, train: bool):
    """The released (bidirectional) H3 on a released window: latents [1,24,37,H,W] at model time model_t."""
    hv = win.video_rows(model, latents)
    t_rows = torch.cat([win.cond_times(), torch.full((hv.shape[1],), float(model_t))])
    tags = torch.cat([win.tags, torch.zeros(hv.shape[1], dtype=torch.int64)])
    hidden, temb, inv = run_blocks(model, torch.cat([win.rows, hv], 1), t_rows, tags,
                                   torch.cat([win.ca_pos, win.target_pos]), train=train)
    return video_output(model, hidden, temb, inv, win.rows.shape[1], latents.shape[2])


def causal_velocity(model, win: Window, latents, model_t: float, train: bool):
    """A streaming DF model on one span (rtcwm.model.objectives.df_forward), every chunk at model time model_t."""
    return df_forward(model, win, latents, [float(model_t)] * SPAN_CHUNKS, {}, train=train)


def noised(x, gen):
    """x at a random teacher-schedule level -> (x_t, n, sigma)."""
    sigma = float(shift(torch.rand(1, generator=gen).clamp(0.02, 0.98), VIDEO_SHIFT))
    n = torch.randn(x.shape, generator=gen).to(x.device)
    return (1 - sigma) * x + sigma * n, n, sigma


def dmd_loss(stack, model, win, x, gen, velocity=released_velocity):
    """DMD generator loss on teacher window x (the chunks that kept their graph carry the gradient)."""
    xt, _, sigma = noised(x.detach(), gen)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        stack.use("teacher")
        x0_real = xt + sigma * velocity(model, win, xt, 1.0 - sigma, train=False).float()
        stack.use("critic")
        x0_fake = xt + sigma * velocity(model, win, xt, 1.0 - sigma, train=False).float()
    p = x.detach() - x0_real
    grad = (x0_fake - x0_real) / p.abs().mean().clamp_min(1e-6)
    stack.use("student")                                         # the graph of x is the student's
    return 0.5 * torch.nn.functional.mse_loss(x.float(), (x.detach() - grad).float()), float(grad.abs().mean())


def critic_loss(stack, model, win, x, gen, velocity=released_velocity):
    """Flow-matching loss of the critic on a detached window."""
    xt, n, sigma = noised(x.detach(), gen)
    stack.use("critic")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = velocity(model, win, xt, 1.0 - sigma, train=True)
    return torch.nn.functional.mse_loss(v.float(), (x.detach() - n).float())


def resume_point(a):
    """(iterations done, student LoRA, critic LoRA): the latest student/critic pair saved in --out (a requeued
    or resubmitted run continues from it -- lowprio jobs are preempted and requeued), else the initial ones."""
    done = sorted(int(p.stem.split("iter")[1]) for p in a.out.glob("student_iter*.safetensors")
                  if (a.out / p.name.replace("student", "critic")).exists())
    if not done:
        return 0, a.student_lora, a.critic_lora
    n = done[-1]
    return n, a.out / f"student_iter{n:05d}.safetensors", a.out / f"critic_iter{n:05d}.safetensors"


def main():
    a = parse()
    causal = a.teacher_kind == "causal"
    velocity = causal_velocity if causal else released_velocity
    w_chunks = SPAN_CHUNKS if causal else TEACHER_CHUNKS            # rollout chunks one teacher window covers
    assert a.grad_chunks <= w_chunks, (a.grad_chunks, w_chunks)
    assert TEACHER_CHUNKS <= a.min_chunks <= a.max_chunks, (a.min_chunks, a.max_chunks)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    log = (lambda *m: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)) if rank == 0 else (lambda *m: None)
    group = node_group(rank, world)
    # NCCL sets up a communicator's buffers (outside the PyTorch allocator) at its first collective. The
    # first cross-node one would be the generator's gradient all-reduce, with the GPU at its fullest: make
    # it now, and keep a few GB out of PyTorch's reach for NCCL.
    torch.cuda.set_per_process_memory_fraction(0.96, dev)
    dist.all_reduce(torch.zeros(1 << 24, device=dev))

    train, val, _ = list_samples(a.val_clips, a.seed, a.picture1)
    starts = Starts(train, a.picture1)
    log(f"distill world={world} train_spans={len(train)} clips={len(starts.spans)} val={len(val)} iters={a.iters} "
        f"student_steps={a.steps} rollout={a.min_chunks}..{a.max_chunks} chunks")
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "val_samples.json").write_text(json.dumps(val))
        (a.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))

    start, student_lora, critic_lora = resume_point(a)
    if start:
        log(f"resuming after iter {start}: {student_lora.name}, {critic_lora.name} (optimizer moments start afresh)")
    tr = loading.load_base()
    stack = loading.LoRAStack(tr, dict(student=(student_lora, True), critic=(critic_lora, True),
                                       teacher=(a.teacher_lora, False)))
    loading.shard_blocks(tr, group, dev)
    stack.to(dev)
    tr.train()
    gen_params, critic_params = stack.params("student"), stack.params("critic")
    reducer = Reducer(gen_params + critic_params)
    opt_g = torch.optim.AdamW(gen_params, lr=a.gen_lr, betas=(0.0, 0.999), weight_decay=0.01)
    opt_c = torch.optim.AdamW(critic_params, lr=a.critic_lr, betas=(0.0, 0.999), weight_decay=0.01)
    warm = lambda i: min(1.0, (i + 1) / a.warmup)                                   # constant after warmup
    sched_g, sched_c = (torch.optim.lr_scheduler.LambdaLR(o, warm) for o in (opt_g, opt_c))
    sigmas = schedule(a.steps, dev, STUDENT_SHIFT)
    history_noise = float(sigmas[-2]) if a.rollout_history == "last-step" else 0.0
    audio = silence()
    gen = torch.Generator().manual_seed(a.seed * 1000 + rank)
    rng = random.Random(a.seed * 1000 + rank)
    log(f"student/critic trainable={sum(p.numel() for p in gen_params) / 1e6:.0f}M each, "
        f"schedule={[round(float(s), 3) for s in sigmas]}")

    def step(opt, sched, params):
        reducer.wait()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        return float(gn)

    t0 = time.time()
    for it in range(start, a.iters):
        # the rollout length and the gradient step fix the sequence of FSDP gathers: equal on every rank
        shared = random.Random(a.seed * 100003 + it)
        n = shared.randint(a.min_chunks, a.max_chunks)
        exit_steps = [shared.randrange(a.steps) for _ in range(n)]
        grad_chunks = list(range(n - a.grad_chunks, n))
        clip, m = starts.draw(rng, n)

        t_it, chunk_times = time.time(), []
        park((opt_g, opt_c), "cpu")
        stack.use("student")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunks = rollout(tr, lambda j: load_window(clip, m + j, audio, start=m, with_target=False, res=a.picture1),
                             n, sigmas, gen, dev, exit_steps, grad_chunks, kv_reuse=a.kv_reuse == "on", history_noise=history_noise,
                             on_chunk=(lambda k, s: chunk_times.append(s)) if it < 3 else None)
        t_roll = time.time() - t_it
        if chunk_times:
            log(f"iter {it + 1} rollout seconds per chunk: {[round(s, 1) for s in chunk_times]}")

        def teacher_window(w):
            """Rollout chunks w .. w+7 as a released window (37 latents) with its conditions. With
            --teacher-picture1 window the window's Picture 1 is its own first frame -- the first latent of the
            rollout's chunk w, which the per-clip VAE makes the frame's own encoding to within a few percent --
            as the released CWM is used (Picture 1 = the window's first frame): the scores then ask for a
            continuation of the scene the rollout is in, not of the start frame seen long ago.
            Causal teacher: rollout chunks w .. w+5 as the student's span m+w (start frame, its caption, causal
            proxy)."""
            if causal:
                sample = load_window(clip, m + w, audio, start=m, with_target=False, res=a.picture1)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    win = make_window(tr, sample, dev, gen)
                return win, torch.cat(chunks[w:w + SPAN_CHUNKS], 2)
            sample = load_window(clip, m + w, audio, start=m, with_target=False, res=a.picture1,
                                 n_latents=TEACHER_LATENTS)
            if a.teacher_picture1 == "window":
                sample["anchor"] = chunks[w][:, :, :1].detach().cpu()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                win = make_window(tr, sample, dev, gen)
            return win, torch.cat(chunks[w:w + TEACHER_CHUNKS], 2)[:, :, :TEACHER_LATENTS]

        win, x = teacher_window(n - w_chunks)
        lg, gmag = dmd_loss(stack, tr, win, x, gen, velocity)
        lg.backward()
        park((opt_g, opt_c), dev)
        gn_g = step(opt_g, sched_g, gen_params)
        del win, x
        chunks = [c.detach() for c in chunks]
        lcs = []
        for _ in range(a.critic_per_gen):
            win, x = teacher_window(rng.randrange(n - w_chunks + 1))
            lc = critic_loss(stack, tr, win, x, gen, velocity)
            lc.backward()
            gn_c = step(opt_c, sched_c, critic_params)
            lcs.append(float(lc))
        stats = torch.tensor([float(lg), sum(lcs) / len(lcs), gmag], device=dev)
        dist.all_reduce(stats, op=dist.ReduceOp.AVG)
        if rank == 0:
            el = time.time() - t0
            log(f"iter {it + 1}/{a.iters} R={n} grad_chunks={len(grad_chunks)} dmd={stats[0]:.4f} critic={stats[1]:.4f} "
                f"|grad|={stats[2]:.3f} gn_g={gn_g:.2e} gn_c={gn_c:.2e} mem={torch.cuda.max_memory_allocated() / 2**30:.0f}G "
                f"rollout {t_roll:.0f}s, iter {time.time() - t_it:.0f}s, eta={(a.iters - it - 1) * el / (it + 1 - start) / 3600:.2f}h")
            with open(a.out / "train_log.jsonl", "a") as fh:
                fh.write(json.dumps(dict(iter=it + 1, chunks=n, exit_steps=exit_steps, grad_chunks=grad_chunks, dmd=float(stats[0]),
                                         critic=float(stats[1]), grad=float(stats[2]), gn_g=gn_g, gn_c=gn_c,
                                         rollout_sec=t_roll, sec=el)) + "\n")
            if (it + 1) % a.save_every == 0 or it + 1 == a.iters:
                for name in ("student", "critic"):
                    p = a.out / f"{name}_iter{it + 1:05d}.safetensors"
                    stack.nets[name].save_weights(str(p), torch.bfloat16, {"base": "cwm", "iter": str(it + 1),
                                                                         "objective": f"dmd-{a.steps}step"})
                log("saved", a.out / f"student_iter{it + 1:05d}.safetensors")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
