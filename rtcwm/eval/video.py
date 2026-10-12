"""Encoding frames / decoding latents with the H3 video VAE, and writing / reading mp4s."""
from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch

from .. import paths


def load_vae(device):
    from musubi_tuner.minimax_h3.video_vae import load_video_vae
    return load_video_vae(paths.VIDEO_VAE, device=device, dtype=torch.float16, disable_mmap=False).eval()


@torch.inference_mode()
def encode_frames(vae, frames: np.ndarray, device) -> torch.Tensor:
    """[frames,768,1344,3] uint8 -> [1,24,F',48,84] latents, as the pre-encoder encodes targets / keyframes."""
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    px = _prepare_pixels(torch.from_numpy(np.ascontiguousarray(frames))).to(device, torch.float16)
    return vae.encode(px).float().cpu()


@torch.inference_mode()
def decode(vae, latents, device) -> np.ndarray:
    """[1,24,F,48,84] latents -> [frames,768,1344,3] uint8."""
    px = vae.decode(latents.to(device, torch.float16))[0].float().cpu()            # [3,frames,H,W] in [-1,1]
    return ((px.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).numpy()


def write_mp4(path: Path, frames: np.ndarray, fps: int = 24, crf: int = 20):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": str(crf)}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def read_mp4(path: Path) -> np.ndarray:
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])
