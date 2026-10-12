"""Pre-encoded windows == the tensors released CWM inference cached for the same clip (bit-exact), and the
clip's latent stream == its first window where they cover the same frames (window 0 starts at frame 0).

The reference cache is CWM inference run on clip_0002 with its coarse proxy
(RTCWM_DATA/cwm_coarse_test/clip_0002_coarse/cache).

    python tests/test_preencode.py
"""
import torch
from safetensors.torch import load_file

from rtcwm import paths

CLIP = "clip_0002"
REF = paths.DATA / "cwm_coarse_test/clip_0002_coarse/cache"


def test_preencode_matches_cwm():
    ours = paths.CACHE / CLIP
    a = load_file(ours / "anchor.safetensors")["latents_ref_000_image"]
    b = load_file(REF / "anchor_reference.safetensors")["latents_ref_000_image"]
    assert torch.equal(a, b)
    ff = load_file(ours / "first_frame.safetensors")["first_frame"]
    assert torch.equal(ff, load_file(REF / "input_video.safetensors")["input_video_latents"][:, :1])
    for w in range(3):
        v = load_file(ours / f"w{w:02d}.visual.safetensors")
        t = load_file(ours / f"w{w:02d}.text.safetensors")
        assert torch.equal(v["duv"], load_file(REF / f"window_{w:03d}/duv_reference.safetensors")["latents_ref_001_video"])
        rq = load_file(REF / f"window_{w:03d}/qwen.safetensors")
        assert torch.equal(t["hidden"], rq["varlen_mmh3_hidden_states_bfloat16"]), w
        assert torch.equal(t["tags"], rq["varlen_mmh3_token_tags_int64"]), w
        assert ("prefix" in v) == (w > 0)
    from rtcwm.data.anchors import cache_paths
    assert torch.equal(load_file(cache_paths(CLIP, 0, "2048")["latent"])["latents_ref_000_image"], a)   # start frame 0 == Picture 1
    stream = load_file(ours / "stream.safetensors")["target"]
    assert torch.equal(stream[:, :35], load_file(ours / "w00.visual.safetensors")["target"][:, :35])


if __name__ == "__main__":
    test_preencode_matches_cwm()
    print("preencode == CWM inference cache; stream == window 0")
