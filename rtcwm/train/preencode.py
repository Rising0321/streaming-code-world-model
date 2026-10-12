"""Pre-encode rendered clips into training windows, exactly as the released CWM inference builds them.

Per clip, in paths.CACHE/<clip>/:
    anchor.safetensors        latents_ref_000_image      Picture 1 (frame 0, short edge 2048)
    first_frame.safetensors   first_frame [24,1,48,84]   frame 0 at 1344x768 (window-0 keyframe)
    wNN.visual.safetensors    target [24,37,48,84]       VAE latent of the window's 124 GT frames
                              duv [24,37,12,21]          coarse-proxy reference latent
                              prefix [24,10,48,84]       (NN>0) GT frames 0..33 re-encoded (Retake34)
    wNN.text.safetensors      hidden [L,5120] bf16, tags [L]   Qwen conditioning (w0 / wn system prompt)
    stream.safetensors        target [24,5K,48,84]       the whole clip as one latent stream: its first K
                                                         17-frame VAE clips (918 of 934 frames), 5 latents each
                              duv [24,5K,12,21]          the coarse proxy, encoded the same way
    anchor_cMM.png / .safetensors, static_cMM.text.safetensors
                              per start frame m (rtcwm.data.anchors): its Picture 1 latent and the Qwen states
                              of its static caption (that frame and its proxy only, w0 system prompt)
paths.CACHE/silence_audio.safetensors   audio [32,2,207]

    python -m rtcwm.train.preencode --stage visual|text|stream|static_text [--shard] [--reverse] [CLIP_DIR ...]
Without CLIP_DIRs every clip of the dataset with a complete prompt set is taken; --shard splits them
over SLURM tasks (task i takes clips i, i+n, ...); --reverse walks from the end (a helper job meeting
the main one in the middle). Existing files are skipped; writes are atomic per process.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import av
import numpy as np
import torch
from safetensors.torch import save_file

from cwm_h3_inference.cache import _load_anchor
from cwm_h3_inference.config import _canonical_caption
from cwm_h3_inference.constants import (AUDIO_LATENT_SHAPE, FPS, OVERLAP_FRAMES, QWEN_OFFSETS, REFERENCE_LATENT_SHAPE,
                                        STRIDE_FRAMES, TARGET_HEIGHT, TARGET_WIDTH, VIDEO_LATENT_SHAPE,
                                        VIDEO_PREFIX_LATENTS, WINDOW_AUDIO_SAMPLES, WINDOW_FRAMES)
from cwm_h3_inference.duv import load_duv_window
from cwm_h3_inference.presentation import build_ref2va_presentation

from .. import paths
from ..data.clip import has_prompts, n_windows, qc_pass, read_meta


def windows_of(clip: Path) -> int:
    return min(n_windows(read_meta(clip)["frames"]), int(os.environ.get("MAX_WINDOWS", 99)))


def read_frames(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def atomic_save(path: Path, tensors: dict, meta: dict | None = None):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")     # unique: a helper job may write the same file
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp), metadata=meta or {})
    tmp.replace(path)


def stage_visual(clips):
    from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition, load_video_vae
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from PIL import Image

    dev = torch.device("cuda:0")
    vae = load_video_vae(paths.VIDEO_VAE, device=dev, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)
    sil = paths.CACHE / "silence_audio.safetensors"
    if not sil.exists():
        avae = load_audio_vae(paths.AUDIO_VAE, device=dev, dtype=torch.float32, disable_mmap=False).eval()
        with torch.inference_mode():
            a = avae.encode(torch.zeros(1, 2, WINDOW_AUDIO_SAMPLES, device=dev)).float().cpu()[0]
        assert tuple(a.shape) == AUDIO_LATENT_SHAPE[1:], a.shape
        paths.CACHE.mkdir(parents=True, exist_ok=True)
        atomic_save(sil, {"audio": a})
        del avae

    def enc(frames_uint8: np.ndarray) -> torch.Tensor:
        px = _prepare_pixels(torch.from_numpy(np.ascontiguousarray(frames_uint8))).to(dev, torch.float16)
        with torch.inference_mode():
            return vae.encode(px).float().cpu()[0]

    for clip in clips:
        out = paths.CACHE / clip.name
        out.mkdir(parents=True, exist_ok=True)
        W = windows_of(clip)
        if all((out / f"w{w:02d}.visual.safetensors").exists() for w in range(W)) and (out / "anchor.safetensors").exists():
            continue
        frames = read_frames(clip / "rgb.mp4")
        assert frames.shape[1:3] == (TARGET_HEIGHT, TARGET_WIDTH), frames.shape
        anchor_png = out / "first_frame.png"
        Image.fromarray(frames[0]).save(anchor_png)
        if not (out / "anchor.safetensors").exists():
            big = _load_anchor(anchor_png)
            px = _prepare_pixels(torch.from_numpy(big).unsqueeze(0)).to(dev, torch.float16)
            with torch.inference_mode():
                lat = encode_video_condition(vae, px)[0].cpu().float()
            atomic_save(out / "anchor.safetensors", {"latents_ref_000_image": lat})
            ff = enc(frames[:1])
            assert ff.shape[1] == 1
            atomic_save(out / "first_frame.safetensors", {"first_frame": ff})
        for w in range(W):
            p = out / f"w{w:02d}.visual.safetensors"
            if p.exists():
                continue
            s = STRIDE_FRAMES * w
            target = enc(frames[s:s + WINDOW_FRAMES])
            assert tuple(target.shape) == VIDEO_LATENT_SHAPE[1:], target.shape
            duv = load_duv_window(clip / "conditions_coarse", s, for_qwen=False)
            dpx = _prepare_pixels(torch.from_numpy(duv.copy())).to(dev, torch.float16)
            with torch.inference_mode():
                ref = encode_video_condition(vae, dpx)[0].cpu().float()
            assert tuple(ref.shape) == REFERENCE_LATENT_SHAPE, ref.shape
            t = {"target": target, "duv": ref}
            if w > 0:
                tail = frames[s:s + OVERLAP_FRAMES]
                padded = np.concatenate([tail, np.repeat(tail[-1:], WINDOW_FRAMES - OVERLAP_FRAMES, 0)], 0)
                t["prefix"] = enc(padded)[:, :VIDEO_PREFIX_LATENTS]
            atomic_save(p, t)
            gc.collect(); torch.cuda.empty_cache()
        print("visual", clip.name, W, "windows", flush=True)


def stage_text(clips):
    from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference
    from musubi_tuner.minimax_h3.text_encoder import (H3TextVisual, build_presentation, encode_h3_presentation,
                                                       load_h3_processor, load_h3_text_encoder)
    dev = torch.device("cuda:0")
    proc = load_h3_processor(str(paths.PROCESSOR))
    model = load_h3_text_encoder(str(paths.TEXT_ENCODER), processor_path=str(paths.PROCESSOR), device=dev,
                                 dtype=torch.bfloat16).eval()
    for clip in clips:
        out = paths.CACHE / clip.name
        anchor_png = out / "first_frame.png"
        anchor = torch.from_numpy(np.ascontiguousarray(_load_anchor(anchor_png))).unsqueeze(0)
        cond = clip / "conditions_coarse"
        for w in range(windows_of(clip)):
            p = out / f"w{w:02d}.text.safetensors"
            if p.exists():
                continue
            caption = _canonical_caption((clip / "prompts" / f"{w:03d}.txt").read_text(), f"{clip.name}/w{w}")
            s = STRIDE_FRAMES * w
            duv = torch.from_numpy(np.ascontiguousarray(load_duv_window(cond, s, for_qwen=True)))
            image_key, video_key = anchor_png, cond / f"{s:06d}.semantic_id.png"
            placeholder = cond / ".h3-presentation-placeholder"
            record = H3Record(video_path=placeholder, caption=caption,
                              target_audio=H3AudioSource(path=placeholder, embedded=False),
                              references=(H3Reference(type="image", path=image_key),
                                          H3Reference(type="video", path=video_key, duration_seconds=WINDOW_FRAMES / FPS)),
                              jsonl_line=0)
            plain = build_presentation(record, "ref2va", {image_key: H3TextVisual(anchor),
                                                          video_key: H3TextVisual(duv, tuple(o / FPS for o in QWEN_OFFSETS))})
            pres = build_ref2va_presentation(proc, plain, w, caption)        # w0 / wn system prompt by index
            with torch.inference_mode():
                hidden, tags = encode_h3_presentation(proc, model, pres)
            atomic_save(p, {"hidden": hidden.to("cpu", torch.bfloat16), "tags": tags.to("cpu", torch.int64)},
                        {"system_prompt": "w0" if w == 0 else "wn"})
            gc.collect(); torch.cuda.empty_cache()
        print("text", clip.name, flush=True)


CLIP_FRAMES = 17                       # the video VAE encodes independent 17-frame clips into 5 latents each


def stage_stream(clips):
    """The whole clip as a continuous latent stream (streaming generation). The VAE encodes 17-frame clips
    independently, so the stream is the concatenation of per-clip latents; one dummy clip is appended
    because the encoder drops the last 3 latents of whatever it is given."""
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition, load_video_vae
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from cwm_h3_inference.duv import load_duv_frame

    dev = torch.device("cuda:0")
    vae = load_video_vae(paths.VIDEO_VAE, device=dev, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)

    def padded(frames: np.ndarray) -> torch.Tensor:
        frames = np.concatenate([frames, np.repeat(frames[-1:], CLIP_FRAMES, 0)], 0)
        return _prepare_pixels(torch.from_numpy(np.ascontiguousarray(frames))).to(dev, torch.float16)

    for clip in clips:
        p = paths.CACHE / clip.name / "stream.safetensors"
        if p.exists():
            continue
        frames = read_frames(clip / "rgb.mp4")
        k = len(frames) // CLIP_FRAMES
        n = k * CLIP_FRAMES
        duv = np.stack([load_duv_frame(clip / "conditions_coarse", i, for_qwen=False) for i in range(n)])
        with torch.inference_mode():
            target = vae.encode(padded(frames[:n])).float().cpu()[0, :, :5 * k]
            ref = encode_video_condition(vae, padded(duv))[0].float().cpu()[:, :5 * k]
        assert target.shape[1] == ref.shape[1] == 5 * k, (target.shape, ref.shape)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(p, {"target": target, "duv": ref}, {"clip_frames": str(CLIP_FRAMES), "vae_clips": str(k)})
        gc.collect(); torch.cuda.empty_cache()
        print("stream", clip.name, k, "VAE clips", flush=True)


def stage_static_text(clips, follow_job: str = "", res: str = "native"):
    """Per start frame (rtcwm.data.anchors): the Picture 1 latent of that frame (short edge 2048, as the
    released CWM encodes its anchor) and the Qwen states of its static caption. The text must not carry
    anything from later frames: Qwen sees Picture 1 and, as the Ref2VA video reference, the proxy of the
    start frame repeated (not a window's proxy video), with the first-window system prompt."""
    from musubi_tuner.minimax_h3.text_encoder import load_h3_processor, load_h3_text_encoder
    from musubi_tuner.minimax_h3.video_vae import load_video_vae
    dev = torch.device("cuda:0")
    vae = load_video_vae(paths.VIDEO_VAE, device=dev, dtype=torch.float16, disable_mmap=False).eval().requires_grad_(False)
    proc = load_h3_processor(str(paths.PROCESSOR))
    model = load_h3_text_encoder(str(paths.TEXT_ENCODER), processor_path=str(paths.PROCESSOR), device=dev,
                                 dtype=torch.bfloat16).eval()
    while True:
        running = bool(follow_job) and _job_running(follow_job)      # captions still being written?
        n = _static_text_pass(clips, vae, proc, model, dev, res)
        if not running:
            break
        if not n:
            time.sleep(60)


def _static_text_pass(clips, vae, proc, model, dev, res: str = "native") -> int:
    from PIL import Image
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from ..data.anchors import anchor_frame, cache_paths, caption_path, clip_anchors
    done = 0
    for clip in clips:
        todo = [m for m in clip_anchors(clip) if caption_path(clip, m).exists() and not cache_paths(clip.name, m, res)["text"].exists()]
        if not todo:
            continue
        frames = read_frames(clip / "rgb.mp4")
        (paths.CACHE / clip.name).mkdir(parents=True, exist_ok=True)
        for m in todo:
            cp = cache_paths(clip.name, m, res)
            Image.fromarray(frames[anchor_frame(m)]).save(cp["png"])
            big = _load_anchor(cp["png"]) if res == "2048" else np.ascontiguousarray(frames[anchor_frame(m)])
            with torch.inference_mode():
                lat = encode_video_condition(vae, _prepare_pixels(torch.from_numpy(big).unsqueeze(0)).to(dev, torch.float16))[0].cpu().float()
            atomic_save(cp["latent"], {"latents_ref_000_image": lat})
            caption = _canonical_caption(caption_path(clip, m).read_text(), f"{clip.name}/static_c{m:02d}")
            hidden, tags = static_text_states(proc, model, big, cp["png"], clip / "conditions_coarse", anchor_frame(m), caption)
            atomic_save(cp["text"], {"hidden": hidden.to("cpu", torch.bfloat16), "tags": tags.to("cpu", torch.int64)},
                        {"system_prompt": "w0", "duv": "start frame repeated", "start_chunk": str(m)})
            gc.collect(); torch.cuda.empty_cache()
        done += len(todo)
        print("static_text", clip.name, len(todo), "start frames", flush=True)
    return done


def static_text_states(proc, model, picture1: np.ndarray, picture1_png: Path, cond: Path, frame: int, caption: str):
    """Qwen states [L,5120] and token tags of a static caption: Qwen sees Picture 1 ([H,W,3] uint8, saved at
    picture1_png) and, as the Ref2VA video reference, the proxy of `frame` repeated -- nothing of later
    frames -- with the first-window system prompt."""
    from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference
    from musubi_tuner.minimax_h3.text_encoder import H3TextVisual, build_presentation, encode_h3_presentation
    from cwm_h3_inference.duv import load_duv_frame
    anchor = torch.from_numpy(np.ascontiguousarray(picture1)).unsqueeze(0)
    f0 = load_duv_frame(cond, frame, for_qwen=True)
    duv = torch.from_numpy(np.ascontiguousarray(np.stack([f0] * len(QWEN_OFFSETS))))
    video_key = cond / f"{frame:06d}.semantic_id.png"
    placeholder = cond / ".h3-presentation-placeholder"
    record = H3Record(video_path=placeholder, caption=caption, target_audio=H3AudioSource(path=placeholder, embedded=False),
                      references=(H3Reference(type="image", path=picture1_png),
                                  H3Reference(type="video", path=video_key, duration_seconds=WINDOW_FRAMES / FPS)),
                      jsonl_line=0)
    plain = build_presentation(record, "ref2va", {picture1_png: H3TextVisual(anchor),
                                                  video_key: H3TextVisual(duv, tuple(o / FPS for o in QWEN_OFFSETS))})
    with torch.inference_mode():
        return encode_h3_presentation(proc, model, build_ref2va_presentation(proc, plain, 0, caption))


def _job_running(job: str) -> bool:
    import subprocess
    return bool(subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True).stdout.strip())


def text_done(clip: Path) -> bool:
    return all((paths.CACHE / clip.name / f"w{w:02d}.text.safetensors").exists() for w in range(windows_of(clip)))


def select_clips(shard: bool, reverse: bool, only_incomplete: bool):
    clips = [d for d in sorted(paths.DATASET.glob("clip_*")) if d.is_dir() and qc_pass(d) and has_prompts(d)]
    if only_incomplete:
        clips = [c for c in clips if not text_done(c)]
    if reverse:
        clips = clips[::-1]
    if shard:
        i, n = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
        clips = clips[i::n]
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["visual", "text", "both", "stream", "static_text"], required=True)
    ap.add_argument("clips", nargs="*", type=Path)
    ap.add_argument("--shard", action="store_true", help="take this SLURM task's share of the dataset")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--only-incomplete", action="store_true", help="skip clips whose text stage is done")
    ap.add_argument("--follow-job", default="", help="static_text: keep encoding new captions while this SLURM job runs")
    ap.add_argument("--picture1", choices=["native", "2048"], default="native", help="static_text: Picture 1 resolution")
    a = ap.parse_args()
    clips = [c.resolve() for c in a.clips] or select_clips(a.shard, a.reverse, a.only_incomplete)
    print(f"{len(clips)} clips", flush=True)
    if a.stage in ("visual", "both"):
        stage_visual(clips)
    if a.stage in ("text", "both"):
        stage_text(clips)
    if a.stage == "stream":
        stage_stream(clips)
    if a.stage == "static_text":
        stage_static_text(clips, a.follow_job, a.picture1)


if __name__ == "__main__":
    main()
