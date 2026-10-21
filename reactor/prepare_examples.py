"""Prepare the official CWM inputs and lightweight browser previews.

The colour MP4 is for preview only; inference uses lossless depth/semantic pairs.
Downloads and expanded conditions live in a user-selected cache. The frontend
links to that cache and to the original reference images without copying them.
"""

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import av
import numpy as np
from PIL import Image
from streaming_cwm_images import PALETTE
from streaming_cwm_inputs import static_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cwm-source", type=Path, help="Pinned code-world-model dependency checkout"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        / "streaming-cwm-examples",
    )
    parser.add_argument(
        "--release-root",
        type=Path,
        help="Reuse already materialized official release inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="External browser cache; defaults to RELEASE_ROOT/reactor-demo",
    )
    args = parser.parse_args()
    gallery = Path(__file__).resolve().parent / "demo/public/examples"
    if gallery.exists() and not gallery.is_symlink():
        parser.error(
            f"{gallery} is a directory. Move its generated contents to an external "
            "cache and remove the empty directory before rerunning with --output."
        )
    release = args.release_root
    if release is None:
        if args.cwm_source is None:
            parser.error("Pass --cwm-source or --release-root")
        from huggingface_hub import snapshot_download

        compact = args.data_dir / "compact"
        snapshot_download(
            repo_id="NTU-yiwen/code-world-model-inference-examples-40",
            repo_type="dataset",
            revision="f8307bf50cb19bed191682734fdf444738160f91",
            local_dir=compact,
        )
        release = args.data_dir / "release"
        subprocess.run(
            [
                sys.executable,
                str(args.cwm_source / "scripts/materialize_hf_examples.py"),
                "--source",
                str(compact),
                "--output",
                str(release),
            ],
            check=True,
        )
    release = release.expanduser().resolve()
    output = (args.output or release / "reactor-demo").expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parent):
        parser.error("Choose a browser cache outside this model workspace")
    media = output / "media"
    media.mkdir(parents=True, exist_ok=True)
    catalog = []
    for example in sorted((release / "examples").glob("example_??")):
        config = json.loads((example / "config.json").read_text())
        prompt = static_prompt(config["windows"][0]["prompt"])
        cond = example / config["condition_root"]
        depth_files = sorted(cond.glob("*.depth.f32"))
        name = example.name
        image = media / f"{name}.png"
        temporary_image = media / f".{name}.{os.getpid()}.png"
        temporary_image.symlink_to((example / config["anchor_image"]).resolve())
        temporary_image.replace(image)
        archive, preview = media / f"{name}.zip", media / f"{name}.mp4"
        if not archive.exists():
            temporary = archive.with_suffix(".zip.tmp")
            with zipfile.ZipFile(
                temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=1
            ) as z:
                for depth in depth_files:
                    z.write(depth, depth.name)
                    semantic = depth.with_name(
                        depth.name.replace(".depth.f32", ".semantic_id.png")
                    )
                    z.write(semantic, semantic.name)
            temporary.replace(archive)
        if not preview.exists():
            temporary = preview.with_suffix(".tmp.mp4")
            with av.open(str(temporary), "w") as container:
                stream = container.add_stream("libx264", rate=24)
                stream.width, stream.height, stream.pix_fmt = 336, 192, "yuv420p"
                stream.options = {"crf": "22"}
                for depth in depth_files:
                    with Image.open(
                        depth.with_name(
                            depth.name.replace(".depth.f32", ".semantic_id.png")
                        )
                    ) as image:
                        frame = av.VideoFrame.from_ndarray(
                            PALETTE[np.asarray(image)], format="rgb24"
                        )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            temporary.replace(preview)
        catalog.append(
            {
                "name": name,
                "label": f"Example {name[-2:]} · {len(depth_files) / 24:.1f}s source",
                "conditions": f"/examples/media/{name}.zip",
                "image": f"/examples/media/{name}.png",
                "preview": f"/examples/media/{name}.mp4",
                "prompt": prompt,
                "max_chunks": 24,
            }
        )
        print(f"Prepared {name}: {len(depth_files)} frames", flush=True)
    if len(catalog) != 40:
        raise ValueError(f"Expected 40 official examples, found {len(catalog)}")
    temporary_index = output / f".index.{os.getpid()}.json"
    temporary_index.write_text(json.dumps(catalog, indent=2))
    temporary_index.replace(output / "index.json")
    gallery.parent.mkdir(parents=True, exist_ok=True)
    temporary_gallery = gallery.with_name(f".examples.{os.getpid()}")
    temporary_gallery.symlink_to(output, target_is_directory=True)
    temporary_gallery.replace(gallery)
    print(f"Browser gallery linked: {gallery} -> {output}", flush=True)


if __name__ == "__main__":
    main()
