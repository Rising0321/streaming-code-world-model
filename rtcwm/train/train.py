"""Causal (streaming) LoRA training of the CWM, initialised from the released CWM LoRA.

    torchrun --nnodes N --nproc-per-node 8 ... -m rtcwm.train.train --out DIR [--steps 150 | --epochs E]

* diffusion forcing on 6-chunk spans cut from the clips' latent streams, each with a drawn start frame
  (rtcwm.train.dataset, rtcwm.model.objectives)
* the frozen H3 blocks are FSDP-sharded over the 8 GPUs of each node; the LoRA weights are replicated
  and their gradients all-reduced asynchronously as each is produced in backward
* one window per GPU per step; steps = epochs * samples / world unless --steps is given
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist

from .. import paths
from ..model import loading
from ..model.layout import SPAN_CHUNKS
from ..model.objectives import LOSSES, OBJECTIVES
from .dataset import StartFrames, list_samples, load_window, silence


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--objective", choices=OBJECTIVES, default="df")
    ap.add_argument("--init-lora", type=Path, default=paths.CWM_LORA)
    ap.add_argument("--steps", type=int, default=0, help="0 = epochs * samples / world")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--val-clips", type=int, default=12)
    ap.add_argument("--max-samples", type=int, default=0, help="smoke tests: cap the training set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--picture1", choices=["native", "2048"], default="native", help="Picture 1 resolution")
    ap.add_argument("--attn", choices=["fa3", "flex"], default="fa3", help="the same mask: FlashAttention-3 prefix calls or flex")
    ap.add_argument("--level-shift", type=float, default=3.0, help="DF: flow shift of the per-chunk noise levels")
    ap.add_argument("--act-offload", choices=["on", "off"], default="off", help="checkpointed block inputs on the CPU")
    return ap.parse_args()


def node_group(rank: int, world: int):
    """One FSDP shard group per node (NVLink); every rank must create every group."""
    lw = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    groups = [dist.new_group(list(range(n * lw, (n + 1) * lw))) for n in range(world // lw)]
    return groups[rank // lw]


def resume_point(a):
    """(steps done, LoRA to start from): the latest checkpoint in --out (a requeued or resubmitted run --
    lowprio jobs are preempted and requeued -- continues from it), else --init-lora."""
    done = sorted(int(p.stem.split("step")[1]) for p in a.out.glob("lora_step*.safetensors"))
    return (done[-1], a.out / f"lora_step{done[-1]:05d}.safetensors") if done else (0, a.init_lora)


def cosine_with_warmup(opt, a, steps):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / a.warmup) * (
        a.lr_min / a.lr + (1 - a.lr_min / a.lr) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps))))


def main():
    a = parse()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    log = (lambda *m: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)) if rank == 0 else (lambda *m: None)
    group = node_group(rank, world)

    train, val, dropped = list_samples(a.val_clips, a.seed, a.picture1)
    if a.max_samples:
        train = train[:a.max_samples]
    steps = a.steps or int(math.ceil(a.epochs * len(train) / world))
    log(f"objective={a.objective} span_chunks={SPAN_CHUNKS} level_shift={a.level_shift} attn={a.attn} act_offload={a.act_offload} world={world} train={len(train)} val={len(val)} dropped_flagged={dropped} steps={steps}")
    if rank == 0:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "val_samples.json").write_text(json.dumps(val))
        (a.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=1))

    tr = loading.load_base()
    start, init = resume_point(a)
    if start:
        log(f"resuming after step {start}: {init.name} (optimizer moments start afresh)")
    net = loading.attach_trainable_lora(tr, init)
    loading.shard_blocks(tr, group, dev)
    net.to(dev, torch.float32)
    tr.train()
    params = list(net.parameters())
    handles = []

    def reduce_hook(p):
        handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))
    for p in params:
        p.requires_grad_(True)
        p.register_post_accumulate_grad_hook(reduce_hook)
    log(f"LoRA trainable={sum(p.numel() for p in params) / 1e6:.1f}M mem_after_shard={torch.cuda.memory_allocated() / 2**30:.1f}G")
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sched = cosine_with_warmup(opt, a, steps)
    for _ in range(start):                       # the schedule where the resumed run left it
        sched.step()
    loss_fn, audio = LOSSES[a.objective], silence()
    offload = a.act_offload == "on"
    gen = torch.Generator().manual_seed(a.seed * 1000 + rank)
    starts = StartFrames(train, a.seed * 1000 + rank, a.picture1)

    order, ptr, mask_cache, t0 = [], 0, {}, time.time()
    for it in range(steps):
        if ptr >= len(order):
            order = train[:]
            random.Random(a.seed + it).shuffle(order)
            order, ptr = order[rank::world], 0
        clip, j = order[ptr]
        ptr += 1
        j, m = starts.draw(clip, j)
        if it < start:                           # resumed: replay the sample order up to where it stopped
            continue
        loss = loss_fn(tr, load_window(clip, j, audio, start=m, res=a.picture1), dev, gen, mask_cache, offload=offload,
                       attn=a.attn, level_shift=a.level_shift)
        loss.backward()
        for h in handles:
            h.wait()
        handles.clear()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        stat = loss.detach().reshape(1)
        dist.all_reduce(stat, op=dist.ReduceOp.AVG)
        if rank == 0:
            el = time.time() - t0
            log(f"step {it + 1}/{steps} loss_v={stat[0]:.4f} gnorm={float(gn):.3f} lr={sched.get_last_lr()[0]:.2e} "
                f"mem={torch.cuda.max_memory_allocated() / 2**30:.0f}G {el / (it + 1 - start):.1f}s/step "
                f"eta={(steps - it - 1) * el / (it + 1 - start) / 3600:.2f}h")
            with open(a.out / "train_log.jsonl", "a") as fh:
                fh.write(json.dumps(dict(step=it + 1, loss_v=float(stat[0]), gnorm=float(gn), lr=sched.get_last_lr()[0],
                                         sec=el)) + "\n")
            if (it + 1) % a.save_every == 0 or it + 1 == steps:
                p = a.out / f"lora_step{it + 1:05d}.safetensors"
                net.save_weights(str(p), torch.bfloat16, {"base": "cwm", "step": str(it + 1), "objective": a.objective,
                                                    "span_chunks": str(SPAN_CHUNKS), "level_shift": str(a.level_shift)})
                log("saved", p)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
