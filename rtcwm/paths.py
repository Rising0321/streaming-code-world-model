"""Every filesystem location the project uses. Override with environment variables.

    RTCWM_DATA    renders, pre-encoded cache, QC reports, runs, eval outputs (default: repo root)
    CWM_RELEASE   the released Code World Model ("release/" of the CWM repo)
    OCWM          open-code-world-model checkout (coarse proxy renderer, rig, follow camera)
    QWEN_VL       Qwen3-VL-32B-Instruct snapshot used for captions
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RTCWM_DATA", REPO))
CWM_RELEASE = Path(os.environ.get("CWM_RELEASE", Path.home() / "projects/code-world-model/release"))
OCWM = Path(os.environ.get("OCWM", Path.home() / "projects/open-code-world-model"))

# --- data generation
CARLA_ROOT = Path(os.environ.get("CARLA_ROOT", DATA / "CARLA_0.9.15"))
DATASET = DATA / "dataset_v1"                      # one directory per rendered clip
QC = DATA / "qc"                                   # per-clip QC reports (clipping/, audit/, coarse/)
DATA_LISTS = REPO / "data_lists"                   # clips kept and windows excluded (in git)

# --- training
CACHE = DATA / "train_cache"                       # pre-encoded windows, one directory per clip
RUNS = DATA / "runs"
EVAL = DATA / "eval"

# --- released models
MODELS = CWM_RELEASE / "models"
DIT = MODELS / "diffusion_models/minimax_h3_ref2va_bf16.safetensors"
CWM_LORA = MODELS / "lora/minimax-h3-cwm.safetensors"
VIDEO_VAE = MODELS / "vae/minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = MODELS / "vae/minimax_h3_audio_vae_fp32.safetensors"
TEXT_ENCODER = MODELS / "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"
PROCESSOR = MODELS / "processors/Qwen3-VL-32B-Instruct"
CWM_EXAMPLES = CWM_RELEASE / "examples"


def qwen_vl() -> Path:
    if "QWEN_VL" in os.environ:
        return Path(os.environ["QWEN_VL"])
    snaps = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots"
    return next(snaps.iterdir())


def use_ocwm() -> None:
    """Make open-code-world-model's `compiler` package importable (it is not pip-installable)."""
    if str(OCWM) not in sys.path:
        sys.path.insert(0, str(OCWM))
