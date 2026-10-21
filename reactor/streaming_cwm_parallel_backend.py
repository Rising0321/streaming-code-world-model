"""Tensor-parallel backend with rank-zero inputs and native VAE tile parallelism."""

import gc
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from streaming_cwm_backend import _Run, prepare_conditioning
from streaming_cwm_checkpoint import load_model
from streaming_cwm_parallel import shard_model
from streaming_cwm_sampling import ChunkSampler

log = logging.getLogger(__name__)


class ParallelBackend:
    def __init__(
        self,
        weights: Path,
        device,
        rank: int,
        world_size: int,
    ):
        from musubi_tuner.minimax_h3.model import Attention
        from rtcwm.model import attention

        torch.set_num_threads(4)
        self.device = torch.device(device)
        self.rank, self.world_size = rank, world_size
        self.weights = weights
        attention.KV_ATTENTION = "sdpa"
        log.info("Rank %d: loading merged CPU weights", rank)
        self.model = load_model(weights)
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.attn_mode = "torch"
        shard_model(self.model, rank, world_size)
        self.model.to(self.device)
        from streaming_cwm_infra import PackedKV, install_adaln_bank, install_fa4
        from streaming_cwm_kernels import install_fused_modulation, install_fused_rope
        from streaming_cwm_prefill import install_prefix_prefill

        log.info("Rank %d: AdaLN bank %s", rank, install_adaln_bank(self.model))
        install_fused_modulation()
        install_fused_rope()
        self.packed_kv = PackedKV(self.model)
        install_fa4()
        install_prefix_prefill(self.model)
        self.vae = self.qwen = self.processor = self.silence = None
        if rank == 0:
            self._load_encoders()
        else:
            from musubi_tuner.minimax_h3.video_vae import load_video_vae

            self.vae = (
                load_video_vae(
                    weights / "vae/minimax_h3_video_vae_fp16.safetensors",
                    device=self.device,
                    dtype=torch.float16,
                    disable_mmap=False,
                )
                .eval()
                .requires_grad_(False)
            )
        from streaming_cwm_infra import (
            GraphTileDecoder,
            install_spatial_decode,
            install_vae_fa4,
        )

        install_vae_fa4(self.vae)
        self.vae.decoder = GraphTileDecoder(self.vae.decoder).eval()
        install_spatial_decode(self.vae, rank, world_size)
        self.reset()
        if world_size > 1:
            dist.barrier()
        log.info(
            "Rank %d: ready, allocated %.2f GiB",
            rank,
            torch.cuda.memory_allocated() / 2**30,
        )

    def _load_encoders(self):
        from cwm_h3_inference.constants import WINDOW_AUDIO_SAMPLES
        from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
        from musubi_tuner.minimax_h3.text_encoder import (
            load_h3_processor,
            load_h3_text_encoder,
        )
        from musubi_tuner.minimax_h3.video_vae import load_video_vae

        weights = self.weights
        audio = load_audio_vae(
            weights / "vae/minimax_h3_audio_vae_fp32.safetensors",
            device=self.device,
            dtype=torch.float32,
            disable_mmap=False,
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
                weights / "vae/minimax_h3_video_vae_fp16.safetensors",
                device=self.device,
                dtype=torch.float16,
                disable_mmap=False,
            )
            .eval()
            .requires_grad_(False)
        )
        processor_path = str(weights / "processors/Qwen3-VL-32B-Instruct")
        self.processor = load_h3_processor(processor_path)
        self.qwen = (
            load_h3_text_encoder(
                str(weights / "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"),
                processor_path=processor_path,
                device="cpu",
                dtype=torch.bfloat16,
            )
            .eval()
            .requires_grad_(False)
        )

    def reset(self):
        self._run: _Run | None = None
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _prepare_on_leader(self, source):
        """Keep the DiT resident when Qwen and its activation workspace fit."""
        offload_dit = torch.cuda.mem_get_info(self.device)[0] < 85 * 2**30
        if offload_dit:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        prepared = prepare_conditioning(
            source, self.vae, self.processor, self.qwen, self.silence, self.device
        )
        if offload_dit:
            self.model.to(self.device)
        return prepared

    @torch.inference_mode()
    def prepare(self, source):
        self.reset()
        from streaming_cwm_infra import GPUBytesDecoder

        # Objects originate only from our trusted rank zero, never from clients.
        payload = [None, None]
        if self.rank == 0:
            try:
                payload[0] = self._prepare_on_leader(source)
            except Exception as error:
                # Every rank must leave the preparation collective together.
                # No failed input is acknowledged and no sampling step follows.
                log.exception("Input preparation failed on rank zero")
                payload[1] = f"{type(error).__name__}: {error}"
        if self.world_size > 1:
            dist.broadcast_object_list(payload, src=0, device=self.device)
        if payload[1] is not None:
            raise RuntimeError(f"Input preparation failed: {payload[1]}")
        prepared = payload[0]
        self._run = _Run(
            sampler=ChunkSampler(
                self.model,
                prepared.window_sample,
                source.chunks,
                source.seed,
                self.device,
            ),
            decoder=GPUBytesDecoder(self.vae),
        )
        log.info("Rank %d: input prepared", self.rank)

    @torch.inference_mode()
    def step(self):
        run = self._run
        if run is None:
            raise RuntimeError("No active render")
        complete = run.sampler.index >= run.sampler.chunks
        if not complete:
            with self.packed_kv.chunk():
                latent = run.sampler.step()
            frames = run.decoder.push(latent)
        else:
            frames = run.decoder.finish()
        run.emitted += int(frames is not None)
        result = frames, run.sampler.index, run.emitted, complete
        if complete:
            self.reset()
        return result
