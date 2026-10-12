"""Window positions == musubi's build_position_grid on a released Ref2VA window (CPU only).

rtcwm.model.packing.window_positions lays out windows of any length the way build_position_grid lays out
the released 37-latent window; on that window both must agree exactly, segment by segment.

    python tests/test_positions.py
"""
import torch
from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_position_grid

from cwm_h3_inference.constants import AUDIO_LATENT_SHAPE, VIDEO_LATENT_SHAPE
from cwm_h3_inference.runtime import build_runtime_batch
from safetensors.torch import load_file

from rtcwm import paths
from rtcwm.model.packing import window_positions

CLIP = "clip_0002"


def test_positions_match_released_layout():
    d = paths.CACHE / CLIP
    v, t = load_file(d / "w01.visual.safetensors"), load_file(d / "w01.text.safetensors")
    anchor = load_file(d / "anchor.safetensors")["latents_ref_000_image"].unsqueeze(0)
    ref = torch.nn.functional.pad(v["duv"].unsqueeze(0), (0, 1, 0, 0))
    rt = build_runtime_batch(anchor, ref, t["hidden"].unsqueeze(0), t["tags"].unsqueeze(0))
    pos = build_position_grid(rt.layout)[0]
    seg = {s.role: s for s in rt.layout.segments}
    ours = window_positions(t["hidden"].shape[0], anchor.shape[3:], H3VideoGeometry(*ref.shape[2:]),
                            AUDIO_LATENT_SHAPE[-1], VIDEO_LATENT_SHAPE[2])
    for name, p in zip(("text", "ref_000_image", "ref_001_video", "target_audio", "target_video"), ours):
        assert torch.equal(p, pos[seg[name].row_slice]), name


if __name__ == "__main__":
    test_positions_match_released_layout()
    print("positions == build_position_grid")
