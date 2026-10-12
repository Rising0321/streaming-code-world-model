"""Stream open-world examples -- the released ones (code-world-model release/examples/example_XX: Picture 1, a
Depth + Semantic-ID proxy, per-window captions) or open-code-world-model takes (examples/gallery.py: a released
first frame, a proxy compiled by the same renderer and rig as our CARLA proxies). There is no ground truth: a
qualitative test outside CARLA.

    python -m rtcwm.eval.open_world prepare --examples DIR/example_01 [...] [--chunks 24]    # one GPU
    python -m rtcwm.eval.open_world generate --lora LORA --out DIR --examples example_01 [...] [--chunks 24]
    python -m rtcwm.eval.open_world release --out DIR --examples example_01 [...] [--chunks 24]

prepare writes each example in the training-cache layout (train_cache/ow_example_XX/), so
rtcwm.train.dataset.load_window streams it like a CARLA clip:
    stream.safetensors {"duv"}   proxy latents of the first --chunks 17-frame clips
    anchorn_c00 / staticn_c00    Picture 1 at 1344x768 (the release's first target frame) and the Qwen states
                                 of the static caption
The static caption is the example's first-window "Appearance and scene" section -- the starting state --
without its action timeline (real time: the text tells nothing of the future).
generate writes <example>_latents.pt and <example>_stream.mp4: generated | proxy (Semantic-ID colours), full
resolution.
release writes DIR/<example>/config.json: the example's own release configuration cut to the windows that
cover --chunks chunks, for the released (bidirectional, windowed) CWM sampler -- python -m
cwm_h3_inference prepare / generate --config ...; it keeps the full per-window captions, timeline included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cwm_h3_inference.constants import TARGET_HEIGHT, TARGET_WIDTH

from .. import paths
from ..data.anchors import cache_paths
from ..model.layout import CLIP_FRAMES

TIMELINE = "Protagonist action timeline:"
PALETTE = np.array([[0, 0, 0], [135, 206, 235], [30, 90, 200], [150, 120, 80], [90, 90, 90], [60, 160, 60],
                    [200, 120, 80], [220, 200, 60], [230, 40, 40], [200, 80, 200], [40, 220, 220], [250, 250, 250]],
                   dtype=np.uint8)                                 # Semantic-ID classes 0..11 (INFERENCE.md)


def cache_name(example: str) -> str:
    return "ow_" + Path(example).name


def static_caption(prompt: str) -> str:
    """A window prompt cut to its "Appearance and scene:" section -- the starting state, no timeline (and no
    instruction preamble, which open-code-world-model's captions carry)."""
    head = "Appearance and scene:"
    return (head + prompt.split(head, 1)[1] if head in prompt else prompt).split(TIMELINE)[0].strip()


def example_inputs(ex: Path):
    """(Picture 1, condition directory, first-window prompt) of a released example (config.json) or of an
    open-code-world-model take (first_frame.png, conditions/, prompts/000.txt)."""
    if (ex / "config.json").exists():
        cfg = json.loads((ex / "config.json").read_text())
        return ex / cfg["anchor_image"], ex / cfg["condition_root"], cfg["windows"][0]["prompt"]
    return ex / "first_frame.png", ex / "conditions", (ex / "prompts" / "000.txt").read_text()


def prepare(examples: list[Path], chunks: int):
    from PIL import Image
    from musubi_tuner.minimax_h3.text_encoder import load_h3_processor, load_h3_text_encoder
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition, load_video_vae
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from cwm_h3_inference.cache import _load_anchor
    from cwm_h3_inference.config import _canonical_caption
    from cwm_h3_inference.duv import load_duv_frame
    from ..train.preencode import atomic_save, static_text_states
    dev = torch.device("cuda:0")
    vae = load_video_vae(paths.VIDEO_VAE, device=dev, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)
    proc = load_h3_processor(str(paths.PROCESSOR))
    qwen = load_h3_text_encoder(str(paths.TEXT_ENCODER), processor_path=str(paths.PROCESSOR), device=dev,
                                dtype=torch.bfloat16).eval()
    for ex in examples:
        anchor, cond, prompt = example_inputs(ex)
        name = cache_name(ex)
        (paths.CACHE / name).mkdir(parents=True, exist_ok=True)
        k = min(chunks, len(list(cond.glob("*.semantic_id.png"))) // CLIP_FRAMES)
        duv = np.stack([load_duv_frame(cond, i, for_qwen=False) for i in range(k * CLIP_FRAMES)])
        duv = np.concatenate([duv, np.repeat(duv[-1:], CLIP_FRAMES, 0)], 0)     # the encoder drops 3 latents
        with torch.inference_mode():
            ref = encode_video_condition(vae, _prepare_pixels(torch.from_numpy(duv)).to(dev, torch.float16))[0]
        atomic_save(paths.CACHE / name / "stream.safetensors", {"duv": ref.float().cpu()[:, :5 * k].contiguous()},
                    {"clip_frames": str(CLIP_FRAMES), "vae_clips": str(k), "source": str(ex)})
        cp = cache_paths(name, 0, "native")
        picture1 = _load_anchor(anchor, size=(TARGET_WIDTH, TARGET_HEIGHT))
        Image.fromarray(picture1).save(cp["png"])
        with torch.inference_mode():
            lat = encode_video_condition(vae, _prepare_pixels(torch.from_numpy(picture1).unsqueeze(0)).to(dev, torch.float16))
        atomic_save(cp["latent"], {"latents_ref_000_image": lat[0].float().cpu()})
        caption = _canonical_caption(static_caption(prompt), f"{name}/static_c00")
        hidden, tags = static_text_states(proc, qwen, picture1, cp["png"], cond, 0, caption)
        atomic_save(cp["text"], {"hidden": hidden.to("cpu", torch.bfloat16), "tags": tags.to("cpu", torch.int64)},
                    {"system_prompt": "w0", "duv": "start frame repeated", "caption": "first window, no timeline"})
        print(f"{name}: {k} chunks, caption {len(caption.split())} words, Qwen rows {hidden.shape[0]}", flush=True)


def proxy_frames(example: Path, n: int) -> np.ndarray:
    from PIL import Image
    cond = example_inputs(example)[1]
    sem = [np.asarray(Image.open(cond / f"{i:06d}.semantic_id.png")) for i in range(n)]
    out = PALETTE[np.clip(np.stack(sem), 0, 11)]
    return np.stack([np.asarray(Image.fromarray(f).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.NEAREST)) for f in out])


def generate(a):
    from cwm_h3_inference.constants import VIDEO_SHIFT
    from ..model.loading import load_for_inference
    from ..model.sampler import generate_stream
    from ..train.dataset import load_window, silence, stream_chunks
    from .video import decode, load_vae, write_mp4
    dev = torch.device("cuda", 0)
    a.out.mkdir(parents=True, exist_ok=True)
    from .evaluate import check_span
    check_span(a.lora)
    from ..model import sampler
    sampler.HISTORY_FROM_LAST_STEP = a.history_from_last_step
    model = load_for_inference(a.lora, dev, fp8=a.fp8)
    audio, results = silence(), {}
    for ex in a.examples:
        name = cache_name(ex)
        n = min(a.chunks, stream_chunks(name))
        lat = generate_stream(model, lambda j: load_window(name, j, audio, start=0, with_target=False), n, a.steps,
                              torch.Generator().manual_seed(a.seed), dev, a.rule, a.shift if a.shift else VIDEO_SHIFT,
                              history_noise=a.history_noise, kv_reuse=a.kv_reuse, on_chunk=lambda k, s: print(f"{name} chunk {k + 1}/{n} {s:.1f}s", flush=True))
        results[ex] = lat.cpu()
    del model
    torch.cuda.empty_cache()
    vae = load_vae(dev)
    for ex, lat in results.items():
        torch.save(lat, a.out / f"{ex}_latents.pt")
        gen = decode(vae, lat, dev)
        write_mp4(a.out / f"{ex}_stream.mp4", np.concatenate([gen, proxy_frames(Path(a.root) / ex, len(gen))], 2), crf=20)
        print("wrote", a.out / f"{ex}_stream.mp4", flush=True)


def release_configs(a):
    from cwm_h3_inference.constants import STRIDE_FRAMES, WINDOW_FRAMES
    windows = 1 + max(0, -(-(a.chunks * CLIP_FRAMES - WINDOW_FRAMES) // STRIDE_FRAMES))
    for e in a.examples:
        ex = Path(a.root) / e
        cfg = json.loads((ex / "config.json").read_text())
        d = (a.out / e).resolve()
        d.mkdir(parents=True, exist_ok=True)
        absolute = lambda v: str((ex / v).resolve())
        cfg.update(anchor_image=absolute(cfg["anchor_image"]), condition_root=absolute(cfg["condition_root"]),
                   models={k: absolute(v) for k, v in cfg["models"].items()}, windows=cfg["windows"][:windows],
                   cache_dir=str(d / "cache"), output=str(d / "output.mp4"), blocks_to_swap=16)
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        print(f"{e}: {windows} windows, {WINDOW_FRAMES + STRIDE_FRAMES * (windows - 1)} frames -> {d / 'config.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "generate", "release"])
    ap.add_argument("--examples", nargs="+", required=True, help="prepare: example directories; generate: their names")
    ap.add_argument("--root", default=str(Path.home() / "projects/code-world-model/release/examples"))
    ap.add_argument("--chunks", type=int, default=24)
    ap.add_argument("--lora", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp8", action="store_true", help="the blocks' large linears in FP8 (Hopper)")
    ap.add_argument("--rule", choices=["euler", "x0"], default="euler", help="x0: few-step (distilled) sampling")
    ap.add_argument("--shift", type=float, default=0.0, help="sampling flow shift (0: the released 12)")
    ap.add_argument("--kv-reuse", action="store_true", help="carry the history's keys/values across chunks")
    ap.add_argument("--history-from-last-step", action=argparse.BooleanOptionalAction, default=True,
                    help="kv-reuse: cache each chunk from its last denoising step (one forward fewer per chunk; "
                         "--no-history-from-last-step: one more clean forward)")
    ap.add_argument("--history-noise", type=float, default=0.0, help="noise level of the history chunks")
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare([Path(e) if Path(e).is_dir() else Path(a.root) / e for e in a.examples], a.chunks)
    elif a.stage == "release":
        release_configs(a)
    else:
        generate(a)


if __name__ == "__main__":
    main()
