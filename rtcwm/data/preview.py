"""Preview media of rendered clips: RGB | proxy | overlay video and a 6-frame strip.

    python -m rtcwm.data.preview OUT_DIR CLIP_DIR [...] [--proxy conditions_coarse]
Writes OUT_DIR/<parent>_<clip>_sbs.mp4, ..._strip.jpg and OUT_DIR/clips.json (one metadata row per clip).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from .clip import FPS, read_meta, read_proxy

PALETTE = np.array([[70, 70, 70], [135, 206, 235], [0, 90, 200], [150, 110, 60], [150, 150, 150], [40, 160, 40],
                    [200, 120, 80], [230, 200, 0], [230, 60, 60], [255, 0, 255], [0, 220, 220], [255, 255, 255]], np.uint8)
W, H = 448, 256


def proxy_image(clip: Path, i: int, kind: str) -> np.ndarray:
    z, s = read_proxy(clip, i, kind)
    dn = 1 - np.log(np.clip(z, 0.3, 256) / 0.3) / np.log(256 / 0.3)
    dn[z <= 1e-3] = 0
    return cv2.resize((0.55 * PALETTE[s] + 0.45 * (dn[..., None] * 255)).astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST)


def label(im, text):
    im = im.copy()
    cv2.putText(im, text, (8, 24), 0, 0.6, (0, 0, 0), 4)
    cv2.putText(im, text, (8, 24), 0, 0.6, (255, 255, 255), 2)
    return im


def preview(clip: Path, out: Path, kind: str = "conditions") -> dict:
    meta = read_meta(clip)
    name = f"{clip.parent.name}_{clip.name}"
    with av.open(str(clip / "rgb.mp4")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    with av.open(str(out / f"{name}_sbs.mp4"), "w") as c:
        st = c.add_stream("libx264", rate=FPS)
        st.width, st.height, st.pix_fmt = 3 * W, H, "yuv420p"
        st.options = {"crf": "26"}
        for i, fr in enumerate(frames):
            r = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            p = proxy_image(clip, i, kind)
            row = np.concatenate([label(r, f"RGB {i / FPS:.1f}s"), label(p, "proxy: depth + semantic"),
                                  label(cv2.addWeighted(r, 0.5, p, 0.5, 0), "overlay")], 1)
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(row), format="rgb24")):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    strip = np.concatenate([label(cv2.resize(frames[i], (W, H)), f"{i / FPS:.1f}s") for i in idx], 1)
    cv2.imwrite(str(out / f"{name}_strip.jpg"), strip[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 82])
    traj = np.array(meta["hero_traj"])
    return dict(name=name, map=meta["map"].replace("_Opt", ""), weather=meta["weather"],
                seconds=round(meta["frames"] / FPS, 1), frames=meta["frames"],
                hero=round(meta["hero_visible_frac_median"] * 100, 1), fps=round(meta["render_fps"], 1),
                end_to_start_m=round(float(np.linalg.norm(traj[-1][:2] - traj[0][:2])), 1),
                path_m=round(float(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1).sum()), 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--proxy", default="conditions", help="conditions (fine) or conditions_coarse")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip in a.clips:
        rows.append(preview(clip, a.out, a.proxy))
        print(rows[-1]["name"], "ok", flush=True)
    (a.out / "clips.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
