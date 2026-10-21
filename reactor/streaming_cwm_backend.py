"""Load the released weights and prepare inputs exactly like rtcwm.eval.open_world.

Qwen is offloaded between renders so the complete BF16 transformer and its
six-chunk cache fit on one B200. Attention uses upstream's SDPA option on
Blackwell; masked prefill retains upstream FlexAttention.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from streaming_cwm_model import RenderSource
from streaming_cwm_sampling import ChunkSampler, StreamingDecoder

logger = logging.getLogger(__name__)


@dataclass
class _Prepared:
    duv: torch.Tensor
    anchor: torch.Tensor
    hidden: torch.Tensor
    tags: torch.Tensor
    silence: torch.Tensor

    def window_sample(self, j: int) -> dict:
        from rtcwm.model.packing import audio_frames

        duv = self.duv[:, j * 5 : j * 5 + 30]
        if duv.shape[1] != 30:
            raise ValueError("Incomplete six-chunk condition span")
        return {
            "ref": torch.nn.functional.pad(duv.unsqueeze(0), (0, 1, 0, 0), value=0.0),
            "anchor": self.anchor,
            "hidden": self.hidden,
            "tags": self.tags,
            "audio": self.silence[..., : audio_frames(30)].unsqueeze(0).clone(),
            "stream_start": j == 0,
        }


@dataclass
class _Run:
    sampler: ChunkSampler
    decoder: StreamingDecoder
    emitted: int = 0


@torch.inference_mode()
def prepare_conditioning(
    source: RenderSource, vae, processor, qwen, silence, device
) -> _Prepared:
    """Shared native image, lossless condition and fixed-caption encoding."""
    from cwm_h3_inference.config import _canonical_caption
    from cwm_h3_inference.duv import load_duv_frame
    from musubi_tuner.minimax_h3.video_vae import encode_video_condition
    from musubi_tuner.minimax_h3_generate_video import _prepare_pixels
    from rtcwm.eval.open_world import static_caption
    from rtcwm.train.preencode import static_text_states

    logger.info("Encoding %d native condition chunks", source.chunks)
    duv = np.stack(
        [
            load_duv_frame(source.conditions, i, for_qwen=False)
            for i in range(source.chunks * 17)
        ]
    )
    duv = np.concatenate((duv, np.repeat(duv[-1:], 17, axis=0)), axis=0)
    pixels = _prepare_pixels(torch.from_numpy(duv)).to(device, torch.float16)
    ref = (
        encode_video_condition(vae, pixels)[0]
        .float()
        .cpu()[:, : source.chunks * 5]
        .contiguous()
    )
    del pixels, duv
    picture = source.image
    anchor = (
        encode_video_condition(
            vae,
            _prepare_pixels(torch.from_numpy(picture).unsqueeze(0)).to(
                device, torch.float16
            ),
        )
        .float()
        .cpu()
    )
    caption = _canonical_caption(static_caption(source.prompt), "reactor/static_c00")
    qwen.to(device)
    try:
        # Native text preparation uses this path only as a reference key;
        # the fitted image pixels are supplied directly in H3TextVisual.
        image_key = source.conditions.parent / "reference.png"
        hidden, tags = static_text_states(
            processor,
            qwen,
            picture,
            image_key,
            source.conditions,
            0,
            caption,
        )
        hidden, tags = hidden.to("cpu", torch.bfloat16), tags.to("cpu", torch.int64)
    finally:
        qwen.to("cpu")
        torch.cuda.empty_cache()
    return _Prepared(
        duv=ref,
        anchor=anchor,
        hidden=hidden.unsqueeze(0),
        tags=tags.unsqueeze(0),
        silence=silence,
    )


class NativeBackend:
    def __init__(self, weights: Path):
        from cwm_h3_inference.constants import WINDOW_AUDIO_SAMPLES
        from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
        from musubi_tuner.minimax_h3.text_encoder import (
            load_h3_processor,
            load_h3_text_encoder,
        )
        from musubi_tuner.minimax_h3.video_vae import load_video_vae
        from rtcwm import paths
        from rtcwm.eval.evaluate import check_span
        from rtcwm.model import attention
        from rtcwm.model.loading import load_for_inference

        self.device = torch.device("cuda", 0)
        weights = Path(weights)
        # These are upstream's process globals; DistributedRunner isolates them.
        paths.DIT = weights / "diffusion_models/minimax_h3_ref2va_bf16.safetensors"
        paths.TEXT_ENCODER = (
            weights / "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
        )
        paths.VIDEO_VAE = weights / "vae/minimax_h3_video_vae_fp16.safetensors"
        paths.AUDIO_VAE = weights / "vae/minimax_h3_audio_vae_fp32.safetensors"
        paths.PROCESSOR = weights / "processors/Qwen3-VL-32B-Instruct"
        lora = weights / "lora/streaming_dmd_4step.safetensors"
        check_span(lora)
        attention.KV_ATTENTION = "sdpa"
        logger.info("Loading native silent audio conditioning")
        audio = load_audio_vae(
            paths.AUDIO_VAE, device=self.device, dtype=torch.float32, disable_mmap=False
        ).eval()
        with torch.inference_mode():
            self.silence = (
                audio.encode(
                    torch.zeros(1, 2, WINDOW_AUDIO_SAMPLES, device=self.device)
                )
                .float()
                .cpu()[0]
            )
        del audio
        self.vae = (
            load_video_vae(
                paths.VIDEO_VAE,
                device=self.device,
                dtype=torch.float16,
                disable_mmap=False,
            )
            .eval()
            .requires_grad_(False)
        )
        self.processor = load_h3_processor(str(paths.PROCESSOR))
        logger.info(
            "Loading Qwen on CPU; it moves to the GPU only for new input preparation"
        )
        self.qwen = (
            load_h3_text_encoder(
                str(paths.TEXT_ENCODER),
                processor_path=str(paths.PROCESSOR),
                device="cpu",
                dtype=torch.bfloat16,
            )
            .eval()
            .requires_grad_(False)
        )
        logger.info("Loading the complete BF16 transformer and distilled LoRA")
        self.model = load_for_inference(lora, self.device, fp8=False)
        from musubi_tuner.minimax_h3.model import Attention

        # rtcwm patches the DiT's KV path, but its token refiner still calls
        # Musubi's original attention. Select native SDPA there as well.
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.attn_mode = "torch"
        self.reset()

    def reset(self) -> None:
        self._run: _Run | None = None

    @torch.inference_mode()
    def prepare(self, source: RenderSource) -> None:
        self.reset()
        self.model.to("cpu")
        torch.cuda.empty_cache()
        prepared = prepare_conditioning(
            source, self.vae, self.processor, self.qwen, self.silence, self.device
        )
        self.model.to(self.device)
        self._run = _Run(
            sampler=ChunkSampler(
                self.model,
                prepared.window_sample,
                source.chunks,
                source.seed,
                self.device,
            ),
            decoder=StreamingDecoder(self.vae),
        )
        logger.info("Input prepared; sampling one native chunk per step")

    @torch.inference_mode()
    def step(self):
        run = self._run
        if run is None:
            raise RuntimeError("No active render")
        sampler, decoder = run.sampler, run.decoder
        if sampler.index < sampler.chunks:
            frames = decoder.push(sampler.step())
            complete = False
        else:
            frames, complete = decoder.finish(), True
        run.emitted += int(frames is not None)
        result = frames, sampler.index, run.emitted, complete
        if complete:
            self.reset()
        return result
