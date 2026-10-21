"""Pinned native checkpoints; no weights are stored in this repository."""

import subprocess
import sys
from pathlib import Path


def load_settings(config_path: Path | None) -> tuple[int, str]:
    """Resolve deployment choices once, before creating any workers."""
    import yaml

    path = config_path or Path(__file__).with_name("streaming_cwm.yaml")
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or set(config) != {"num_gpus", "inference_mode"}:
        raise ValueError("Configuration requires num_gpus and inference_mode")
    count, mode = config["num_gpus"], config["inference_mode"]
    if type(count) is not int or count not in (1, 4, 8):
        raise ValueError("num_gpus must be 1, 4 or 8")
    if mode not in ("native", "fa4"):
        raise ValueError("inference_mode must be native or fa4")
    if mode == "native" and count != 1:
        raise ValueError("The native reference path requires num_gpus: 1")
    return count, mode


ASSETS = (
    (
        "Luffuly/streaming-code-world-model",
        "a9affd262d83907896f80bd08ee89a5d19116ead",
        "lora",
        ["streaming_dmd_4step.safetensors", "README.md"],
    ),
    (
        "Comfy-Org/MiniMax-H3",
        "0543966fbdce5ba05709a8f2031c94bdba629b4a",
        "",
        [
            "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ],
    ),
    (
        "Qwen/Qwen3-VL-32B-Instruct",
        "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        "processors/Qwen3-VL-32B-Instruct",
        [
            "*.json",
            "*.txt",
            "*.model",
            "*.tiktoken",
            "README.md",
            "LICENSE*",
        ],
    ),
)


def download_weights(root: Path) -> None:
    from huggingface_hub import snapshot_download

    for repo, revision, subdir, patterns in ASSETS:
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=root / subdir,
            allow_patterns=patterns,
        )


def prepare_demo_inputs(root: Path, cwm_source: Path) -> Path:
    """Materialize the pinned official depth/ID data once beside the weights."""
    from huggingface_hub import snapshot_download

    examples = root / "examples"
    if (examples / "materialization.json").is_file() and all(
        (examples / f"example_{i:02d}/config.json").is_file() for i in range(1, 41)
    ):
        return examples
    compact = root / "example-download"
    snapshot_download(
        repo_id="NTU-yiwen/code-world-model-inference-examples-40",
        repo_type="dataset",
        revision="f8307bf50cb19bed191682734fdf444738160f91",
        local_dir=compact,
    )
    subprocess.run(
        [
            sys.executable,
            str(cwm_source / "scripts/materialize_hf_examples.py"),
            "--source",
            str(compact),
            "--output",
            str(root),
        ],
        check=True,
    )
    return examples
