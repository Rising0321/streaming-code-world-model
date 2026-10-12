"""CWM-style captions with Qwen3-VL-32B: per window (the released CWM format) or one static caption per clip.

Each 124-frame window (stride 90) gets 8 evenly spaced frames, one released CWM prompt as the format
reference, and the ground-truth motion of the protagonist per 0.75 s slot (from the simulated
trajectory, so a stop or a turn-round hard to see in 8 frames is not missed). Windows after the first
get CWM's rolling-history preamble. Output: CLIP/prompts/NNN.txt; existing prompts are kept.

    python -m rtcwm.data.captions CLIP_DIR [CLIP_DIR ...]      (env MAX_WINDOWS limits windows per clip)
    python -m rtcwm.data.captions --static CLIP_DIR [...]   -> CLIP/prompts/static_cMM.txt

Static captions are for streaming generation: a stream starts from a start frame (Picture 1), and the
caption describes only that frame -- the protagonist and the scene visible in it -- plus one fixed sentence
on the walking and the trailing camera, so the text carries no timing and nothing from later frames;
motion comes from the proxy alone. Start frames: the first frame of every ANCHOR_EVERY-th chunk
(rtcwm.data.anchors), so training draws Picture 1 from many points of a clip.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import av
import numpy as np

from .. import paths
from .clip import FPS, STRIDE_FRAMES, WINDOW_FRAMES, n_windows, read_meta

REFERENCE_EXAMPLE = "example_30"
HISTORY = ("The first 1.4167 seconds are the immutable 34-frame rolling history and remain unchanged, continuing the "
           "established camera motion, positions, scale, orientation, materials, lighting, and ongoing actions without a cut, restart, or reset.\n\n")
SLOTS = [(0.0, 0.75), (0.75, 1.5), (1.5, 2.25), (2.25, 3.0), (3.0, 3.75), (3.75, 4.5), (4.5, 5.17)]
N_FRAMES = 8

INSTRUCTION = """You write video captions for a world model. Below is a REFERENCE caption that shows the required format and level of detail.

REFERENCE:
<<<
{ref}
>>>

Now look at the {n} frames, which are evenly spaced over a 5.17-second clip (frame k is at {times}).
Write a caption for THIS clip in exactly the same structure as the reference:
1. "Appearance and scene:" then one paragraph describing the protagonist (the person the camera follows, seen mostly from behind): build, hair, clothing, stable identity features, and where they are in frame at 0.00s.
2. One paragraph describing the setting: ground surface, road, buildings, vegetation, vehicles, sky, lighting and weather, and the camera (a close trailing third-person camera behind the protagonist) and how it moves.
3. "Protagonist action timeline:" then exactly seven segments "0.00–0.75s: ...; 0.75–1.50s: ...; 1.50–2.25s: ...; 2.25–3.00s: ...; 3.00–3.75s: ...; 3.75–4.50s: ...; 4.50–5.17s: ..." describing walking, turning left or right, slowing, crossing, etc., based on what changes between frames.
4. "Other visible actors:" then a short paragraph about other cars or people that move, with time ranges, or state that none move.
Ground-truth motion of the protagonist in this clip (follow it exactly in the action timeline; the trailing camera always swings round to stay behind the protagonist, so after turning around the protagonist is again seen from behind):
{hints}
Describe only what is visible. Do not mention frames, images, renders, simulation, CARLA, or games. Output the caption only."""


def reference_prompt() -> str:
    cfg = json.loads((paths.CWM_EXAMPLES / REFERENCE_EXAMPLE / "config.json").read_text())
    return cfg["windows"][1]["prompt"].replace("\r\n", "\n")


def motion_hints(traj, w: int) -> str:
    t = np.asarray(traj)[:, :2]
    head = []
    for i in range(len(t)):
        d = t[min(len(t) - 1, i + 6)] - t[max(0, i - 6)]
        head.append(math.degrees(math.atan2(d[1], d[0])) if np.hypot(*d) > 0.3 else None)
    out = []
    for s0, s1 in SLOTS:
        f0, f1 = STRIDE_FRAMES * w + int(s0 * FPS), min(STRIDE_FRAMES * w + int(s1 * FPS), len(t) - 1)
        dist = float(np.linalg.norm(t[f1] - t[f0]))
        h0, h1 = head[f0], head[f1]
        turn = None if h0 is None or h1 is None else abs((h1 - h0 + 180) % 360 - 180)
        if dist < 0.25:
            state = "stands still / pauses"
        elif turn is not None and turn > 100:
            state = "turns around and starts walking back the way they came"
        elif turn is not None and turn > 35:
            state = "walks while turning to follow the path"
        else:
            state = "walks steadily forward along the path"
        out.append(f"{s0:.2f}-{s1:.2f}s: {state}")
    return "; ".join(out)


def clean_caption(text: str) -> str:
    """Strip prompt delimiters the model sometimes echoes; put the seven timeline segments on one
    line joined by '; ' as in the released CWM captions."""
    text = text.replace("<<<", "").replace(">>>", "").strip()
    if "Protagonist action timeline:" in text:
        head, rest = text.split("Protagonist action timeline:", 1)
        tl, sep, tail = rest.partition("Other visible actors:")
        segs = [re.sub(r"\s+", " ", x).strip().rstrip(";") for x in re.split(r"\n\s*\n|;\s*(?=\d+\.\d+[–-])", tl) if x.strip()]
        text = head.rstrip() + "\n\nProtagonist action timeline:\n\n" + "; ".join(segs)
        if sep:
            text += "\n\nOther visible actors:\n\n" + re.sub(r"\s+", " ", tail).strip()
    return text


STATIC_INSTRUCTION = """You write the text prompt for a world model that continues a video from this single starting image.
Write exactly two paragraphs, starting with the line "Appearance and scene:":
1. The protagonist (the person in the foreground, seen from behind): build, hair, skin tone, clothing, shoes and stable
   identity features, and where they are in the frame.
2. The setting visible in this image: ground surface, sidewalk, road, buildings, walls, vegetation, vehicles, other
   people, sky, lighting and weather, and the camera (a close trailing third-person camera behind the protagonist).
Describe only what is visible in this image. Do not describe actions, events or anything that happens later. Do not
mention images, renders, simulation, CARLA, or games. Output the two paragraphs only."""
STATIC_MOTION = ("The protagonist walks along the sidewalk, and the close trailing third-person camera follows behind "
                 "them, swinging round to stay behind the protagonist whenever they turn.")


class Captioner:
    def __init__(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(str(paths.qwen_vl()))
        self.model = AutoModelForImageTextToText.from_pretrained(str(paths.qwen_vl()), torch_dtype=torch.bfloat16,
                                                                 device_map="auto").eval()
        self.ref = reference_prompt()

    def caption(self, images, times: str, hints: str) -> str:
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": INSTRUCTION.format(ref=self.ref, n=len(images), times=times, hints=hints)})
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                               add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=1200, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def caption_static(self, clip: Path):
        from .anchors import anchor_frame, caption_path, clip_anchors
        todo = [m for m in clip_anchors(clip) if not caption_path(clip, m).exists()]
        if not todo:
            return
        with av.open(str(clip / "rgb.mp4")) as c:
            frames = {}
            for i, f in enumerate(c.decode(video=0)):
                if i in {anchor_frame(m) for m in todo}:
                    frames[i] = f.to_image()
        for m in todo:
            self._caption_frame(frames[anchor_frame(m)], caption_path(clip, m))
            print(clip.name, "static", m, flush=True)

    def _caption_frame(self, first, outp: Path):
        content = [{"type": "image", "image": first}, {"type": "text", "text": STATIC_INSTRUCTION}]
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                               add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=700, do_sample=False)
        text = self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        text = text.replace("<<<", "").replace(">>>", "").strip()
        outp.parent.mkdir(exist_ok=True)
        outp.write_text(text + "\n\n" + STATIC_MOTION)

    def caption_clip(self, clip: Path):
        meta = read_meta(clip)
        windows = min(n_windows(meta["frames"]), int(os.environ.get("MAX_WINDOWS", 99)))
        (clip / "prompts").mkdir(exist_ok=True)
        with av.open(str(clip / "rgb.mp4")) as c:
            video = [f.to_image() for f in c.decode(video=0)]
        for w in range(windows):
            outp = clip / "prompts" / f"{w:03d}.txt"
            if outp.exists():
                continue
            idx = [STRIDE_FRAMES * w + round(k * (WINDOW_FRAMES - 1) / (N_FRAMES - 1)) for k in range(N_FRAMES)]
            times = ", ".join(f"{k}: {(i - STRIDE_FRAMES * w) / FPS:.2f}s" for k, i in enumerate(idx))
            text = clean_caption(self.caption([video[i].resize((672, 384)) for i in idx], times,
                                              motion_hints(meta["hero_traj"], w)))
            outp.write_text((HISTORY + text) if w > 0 else text)
            print(clip.name, w, len(text.split()), "words", flush=True)


def main():
    args = sys.argv[1:]
    static = "--static" in args
    cap = Captioner()
    for clip in map(Path, [a for a in args if a != "--static"]):
        cap.caption_static(clip) if static else cap.caption_clip(clip)


if __name__ == "__main__":
    main()
