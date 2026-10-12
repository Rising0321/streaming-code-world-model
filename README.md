<div align="center">

<h2>Streaming Code World Model</h2>

<div align="center">
  <a href='https://github.com/buaacyw/code-world-model'><img src='https://img.shields.io/badge/built_on-Code_World_Model-purple?logo=github&logoColor=black&labelColor=white'></a> &ensp;
  <a href='https://huggingface.co/Luffuly/streaming-code-world-model'><img src='https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-checkpoints-yellow?labelColor=white'></a>
</div>

</div>

https://github.com/user-attachments/assets/37da6610-90eb-4174-9f16-fa5e0b628b1c

<div align="center"><sub>4-step streaming generation; proxy inset bottom-right.</sub></div>

---

This project implements a streaming version of [Code World Model](https://github.com/buaacyw/code-world-model) (CWM).
Each 0.71 s chunk is generated causally from the start image, the proxy so far and the previous chunks, with no access
to the future, and a 4-step distilled model renders a chunk in 5.1 s on one H200.

## ✨ Highlights

- **Causal streaming** — one chunk per VAE clip, a sliding 6-chunk attention span, KV cache moved by RoPE rotation.
- **Diffusion forcing with a noisy history** — slows long-horizon drift.
- **4-step DMD self forcing** — 9× faster than 35-step sampling, and more accurate.
- **Fast** — FlashAttention-3 in training and sampling, optional FP8; 5.1 s per chunk on one H200 (not yet real time).

## 📊 Results

Four held-out CARLA clips, 24 chunks (17 s) each; pixel MSE to ground truth (lower is better), one seed.
Inference runs on **one NVIDIA H200 (141 GB)** at 1344×768; peak GPU memory is **123 GB** in bf16 and **108 GB** with
`--fp8` (nvidia-smi peak, GiB); 80 GB GPUs are not supported yet.

| model | steps | mean | 0–4 s | 13–17 s | s / chunk |
|:---|:---:|---:|---:|---:|---:|
| streaming diffusion forcing | 35 | 2231 | 1335 | 3332 | 47.9 |
| ↳ + DMD, released CWM as teacher | 4 | 1741 | **1049** | **2501** | 5.2 |
| **↳ + DMD, streaming model as teacher** | **4** | **1660** | 1070 | 2555 | **5.1** |

## 🔧 Installation

```bash
git clone https://github.com/TingtingLiao/streaming-code-world-model.git && cd streaming-code-world-model
```

- **`cwm` env** — PyTorch ≥ 2.9 (CUDA 12.8), FlashAttention-3, `transformers`, and from the
  [CWM repository](https://github.com/buaacyw/code-world-model) its patched musubi-tuner, `cwm_h3_inference` and the
  released models.
- **`carla` env** (data only) — CARLA 0.9.15.
- [open-code-world-model](https://github.com/guangyliu/cwm-proxy) and Qwen3-VL-32B-Instruct (captions).

```bash
export RTCWM_DATA=/path/to/data CWM_RELEASE=/path/to/code-world-model/release
export OCWM=/path/to/open-code-world-model QWEN_VL=/path/to/Qwen3-VL-32B-Instruct
source scripts/env.sh
```

## 🚀 Sampling

Checkpoints (LoRAs for the released MiniMax-H3) are on
[Hugging Face](https://huggingface.co/Luffuly/streaming-code-world-model): `streaming_df.safetensors` (35 steps) and
`streaming_dmd_4step.safetensors` (4 steps).

```bash
hf download Luffuly/streaming-code-world-model --local-dir ckpt

# held-out CARLA clips (generated | ground truth, per-chunk metrics)
EVAL_ARGS="--steps 4 --rule x0 --shift 5 --kv-reuse" \
    sbatch scripts/eval/eval.sbatch ckpt/streaming_dmd_4step.safetensors out/ clip_0048 clip_0244

# open world: open-code-world-model scenes, 60 s
WINDOWS=16 OW_ARGS="--chunks 85 --steps 4 --rule x0 --shift 5 --kv-reuse" \
    sbatch scripts/eval/openworld_long.sbatch scenes/ out/ ckpt/streaming_dmd_4step.safetensors
```

`--fp8` runs the large linears in FP8. By default each chunk enters the history from its last denoising step (one
forward fewer, and the noisy history keeps long streams from drifting); `--no-history-from-last-step` uses a clean
forward instead.

## 🏋️ Training

```bash
# 1. data (carla env): render, coarse proxy, QC, start-frame captions
python -m rtcwm.data.manifest --n 400 --seed0 30000 > $RTCWM_DATA/manifest.jsonl
sbatch scripts/data/render_dataset.sbatch $RTCWM_DATA/manifest.jsonl $RTCWM_DATA/dataset_v1
sbatch scripts/data/qc.sbatch $RTCWM_DATA/dataset_v1
CAPTION_ARGS=--static sbatch scripts/data/caption.sbatch $RTCWM_DATA/dataset_v1

# 2. pre-encode and smoke test (cwm env)
sbatch scripts/train/preencode.sbatch --stage stream
sbatch scripts/train/preencode.sbatch --stage static_text
sbatch scripts/train/smoke.sbatch

# 3. streaming diffusion forcing (8 nodes, ~7 h)
sbatch scripts/train/train.sbatch $RTCWM_DATA/runs/df --init-lora $CWM_RELEASE/models/lora/minimax-h3-cwm.safetensors \
    --epochs 5 --level-shift 3

# 4. 4-step DMD self forcing, the streaming model as teacher (4 nodes, ~5 h)
DF=$RTCWM_DATA/runs/df
sbatch scripts/train/distill.sbatch $RTCWM_DATA/runs/dmd $DF/lora_step00800.safetensors --iters 100 \
    --teacher-kind causal --teacher-lora $DF/lora_step01118.safetensors --critic-lora $DF/lora_step01118.safetensors
```

Both trainings resume from the latest checkpoint in their output directory. The Slurm scripts assume 8-GPU nodes.

<details>
<summary><b>How it works</b></summary>

<br>

- **Chunks.** One chunk = one 17-frame VAE clip (5 latents). Chunk *k* is generated inside its span (itself and the
  5 chunks before it), with positions counted from the span's first chunk, so they never grow with the stream.
- **Causal conditioning.** Video chunk *c* sees the caption, the start image, proxy chunks ≤ *c* and earlier video
  chunks of its span. The caption describes the start image only.
- **Diffusion forcing.** Independent per-chunk noise levels, so the model learns not to trust its history blindly.
- **DMD self forcing.** The student streams 8–14 chunks on its own output; a frozen teacher scores the last span and
  a critic, initialised from it, tracks the student.
- **Data.** 342 CARLA clips (3.7 h): a pedestrian followed by a third-person camera, with coarse proxies from
  open-code-world-model.

</details>

## 🙏 Acknowledgements

The proxy code was built by [Guangyi Liu](https://github.com/guangyliu). Built on [Code World Model](https://github.com/buaacyw/code-world-model), MiniMax-H3,
[musubi-tuner](https://github.com/kohya-ss/musubi-tuner), [CARLA](https://carla.org) and Qwen3-VL. Causal training and
distillation follow diffusion forcing, CausVid and Self Forcing.

## 📖 Citation

```bibtex
@misc{liao2026streamingcwm,
  title        = {Streaming Code World Model},
  author       = {Liao, Tingting and Liu, Guangyi},
  year         = {2026},
  howpublished = {\url{https://github.com/TingtingLiao/streaming-code-world-model}}
}
```
