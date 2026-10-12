"""Causal CWM: the released MiniMax-H3 Ref2VA model + CWM LoRA, run chunk-causally.

A 124-frame CWM window (37 latents) is packed as
    condition : text | Picture 1 anchor | Video 1 coarse-proxy DUV | keyframe (target latent 0)
    audio     : clean silence (CARLA is silent)
    video     : target latents 1..36 = 7 chunks of 6, 5, 5, 5, 5, 5, 5 latents (each generated prefix is 5n+2
                latents, aligned with the VAE's 17-frame clips; see rtcwm.model.layout)
Two training objectives (rtcwm.model.objectives):
    df : diffusion forcing -- one copy of the video, every chunk at its own noise level; the condition,
         audio and chunk 0 see each other, chunk c >= 1 sees the condition, audio and chunks c-5..c.
         Condition rows have fixed (clean) times, so a sampler caches their keys/values together with
         chunk 0's (rtcwm.model.sampler.KVSampler).
    tf : teacher forcing (SolarWM Stage1 layout) -- clean and noisy copies of the video; noisy chunk c
         sees the condition, audio, clean chunks c-5..c-1 and itself; condition times follow the
         video time as in the released CWM.
"""
