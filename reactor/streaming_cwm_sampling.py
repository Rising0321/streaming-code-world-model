"""Suspend the released KV-reuse sampler at native chunk boundaries.

The algorithm is rtcwm.model.sampler._stream's inference branch (4 x0 steps,
shift 5, six chunks, history from the last step). Native helpers implement all
attention, condition augmentation, RoPE and cache operations. No GPU generator,
extra random draw, reduced context, quantization or shortened schedule is used.
"""

import torch


class ChunkSampler:
    def __init__(self, model, window_sample, chunks: int, seed: int, device):
        from musubi_tuner.minimax_h3.packing import _video_t_spans
        from rtcwm.model import sampler
        from rtcwm.model.layout import SPAN_CHUNKS, chunk_shape

        if SPAN_CHUNKS != 6:
            raise ValueError(
                "The released distilled checkpoint requires a six-chunk span"
            )
        if not (sampler.HISTORY_FROM_LAST_STEP and sampler.ROTATE_IN_ATTENTION):
            raise ValueError("Use the released inference cache defaults")
        if sampler.PROXY_ONLY_CONDITIONS:
            raise ValueError("Experimental proxy-only conditioning is not supported")
        self.model, self.window_sample = model, window_sample
        self.chunks, self.device = chunks, device
        self.gen = torch.Generator().manual_seed(seed)
        self.sigmas = sampler.schedule(4, device, 5.0)
        self.chunk_span = sum(_video_t_spans(chunk_shape()[2]))
        self.index = 0
        self.history = {}
        self.conditions = self.static = self.previous_start = None
        self.tables = {}

    @torch.inference_mode()
    def step(self):
        from rtcwm.model import sampler as native
        from rtcwm.model.layout import chunk_shape, span_start

        k = self.index
        if k >= self.chunks:
            raise RuntimeError("The latent stream is complete")
        j = span_start(k)
        # make_window consumes RNG for both visual conditions on EVERY chunk.
        win = native.make_window(
            self.model, self.window_sample(j), self.device, self.gen
        )
        kvs = native.KVSampler(self.model, win, train=False)
        if self.previous_start is not None and j != self.previous_start:
            delta = -(j - self.previous_start) * self.chunk_span
            self.history = {
                i: native._move(self.model, kv, delta, self.device)
                for i, kv in self.history.items()
                if i >= j
            }
            self.conditions = None
        if k > j and self.conditions is None:
            kvs.add_conditions(None)
            self.conditions = kvs.ca_kv
            self.static = native._share_static(
                self.conditions, self.static, win.static_sees_first
            )
        kvs.ca_kv = self.conditions
        kvs.chunks_kv = [
            native._moved(self.model, *self.history[i], self.device, self.tables)
            for i in range(j, k)
        ]
        noise = torch.randn(chunk_shape(), generator=self.gen).to(self.device)
        last_kv = k > j and k < self.chunks - 1
        x = kvs.generate_chunk(
            k - j,
            noise,
            self.sigmas,
            "x0",
            self.gen,
            cache=False,
            keep_last_kv=last_kv,
        ).float()
        if k < self.chunks - 1:
            if k == j:
                kvs.add_chunk(0, x.detach(), model_t=1.0)
                self.conditions = kvs.ca_kv
                self.history[k] = [kvs.chunks_kv[0], 0.0]
                self.static = native._share_static(
                    self.conditions, self.static, win.static_sees_first
                )
            else:
                self.history[k] = [kvs.last_kv, 0.0]
        self.previous_start = j
        self.index += 1
        return x


class StreamingDecoder:
    """Native VAE clip stitching with one chunk of lookahead.

    Five latent frames produce 17 output frames, but the decoder reads two latent
    frames from the NEXT chunk. Keep that lookahead and the native five-frame blend
    instead of independently decoding/padding every chunk. EOF repeats the final
    latent exactly as MiniMaxH3VideoVAE._decode_video does.
    """

    def __init__(self, vae):
        self.vae = vae
        self.pending = self.overlap = None

    @torch.inference_mode()
    def push(self, latent):
        if self.pending is None:
            self.pending = latent
            return None
        frames = self._decode(torch.cat((self.pending, latent[:, :, :2]), dim=2))
        self.pending = latent
        return frames

    @torch.inference_mode()
    def finish(self):
        if self.pending is None:
            raise RuntimeError("No pending latent chunk")
        frames = self._decode(
            torch.cat(
                (self.pending, self.pending[:, :, -1:].repeat(1, 1, 2, 1, 1)), dim=2
            )
        )
        self.pending = self.overlap = None
        return frames

    def _decode(self, latent):
        vae = self.vae
        latent = latent.to(dtype=torch.float16)
        latent = latent * vae.latents_std.view(1, -1, 1, 1, 1).to(
            latent
        ) + vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        clip = vae._decode_clip(latent)
        frames = clip[:, :, 3:20]
        if self.overlap is not None:
            frames = vae._blend(self.overlap, frames, 5, dim=-3)
        self.overlap = clip[:, :, 23:].clone()
        pixels = frames.float() * vae.pixel_std.to(frames) + vae.pixel_mean.to(frames)
        pixels = pixels.clamp(0.0, 1.0) * 2.0 - 1.0
        pixels = pixels[0].float().cpu()
        return (
            ((pixels.clamp(-1, 1) + 1) * 127.5)
            .round()
            .byte()
            .permute(1, 2, 3, 0)
            .contiguous()
            .numpy()
        )
