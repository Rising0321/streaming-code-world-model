"""Strict memory-mapped loading; preserve native BF16 weights and LoRA merge."""

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file


def load_model(weights):
    from cwm_h3_inference.runtime import strict_merge_lora
    from musubi_tuner.minimax_h3.model import (
        MiniMaxH3Model,
        parse_h3_transformer_config,
    )
    from rtcwm.model import attention

    path = weights / "diffusion_models/minimax_h3_ref2va_bf16.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as source:
        config = parse_h3_transformer_config(source.metadata())
    with init_empty_weights():
        model = MiniMaxH3Model(
            config, attn_mode="torch", split_attn=False, dtype=torch.bfloat16
        )
    state = load_file(str(path), device="cpu")
    expected = model.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("Checkpoint keys differ from the native model")
    for name, value in state.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    del state, expected
    strict_merge_lora(model, weights / "lora/streaming_dmd_4step.safetensors", 200)
    attention.install(model)
    return model
