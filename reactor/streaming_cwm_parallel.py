"""Tensor parallelism over complete H3 attention heads and gated MLP pairs.

Split attention by complete heads and gated MLPs by complete gate/value pairs.
Keep all sequence positions on each rank, so native masks, KV eviction and RoPE
positions retain their meaning. BF16 all-reduce changes accumulation order;
results across GPU counts need not be bit-identical.
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class _RowLinear(nn.Linear):
    """Each rank owns input columns; sum partial outputs before adding bias."""

    def forward(self, value):
        output = F.linear(value, self.weight)
        dist.all_reduce(output, group=self.group)
        if self.bias is not None:
            output = output + self.bias
        return output


def _column_slice(layer, rank, world_size, groups):
    if layer.out_features % (groups * world_size):
        raise ValueError("Each fused projection group must divide across ranks")
    width = layer.out_features // groups
    local = width // world_size
    weight = torch.cat(
        [
            layer.weight[g * width + rank * local : g * width + (rank + 1) * local]
            for g in range(groups)
        ],
        dim=0,
    )
    result = nn.Linear(
        layer.in_features,
        local * groups,
        bias=layer.bias is not None,
        dtype=layer.weight.dtype,
        device="meta",
    )
    result.weight = nn.Parameter(weight, requires_grad=False)
    if layer.bias is not None:
        result.bias = nn.Parameter(
            torch.cat(
                [
                    layer.bias[
                        g * width + rank * local : g * width + (rank + 1) * local
                    ]
                    for g in range(groups)
                ],
                dim=0,
            ),
            requires_grad=False,
        )
    return result


def _row_slice(layer, rank, world_size, group):
    if layer.in_features % world_size:
        raise ValueError("Input projection width must divide across ranks")
    width = layer.in_features // world_size
    result = _RowLinear(
        width,
        layer.out_features,
        bias=layer.bias is not None,
        dtype=layer.weight.dtype,
        device="meta",
    )
    result.weight = nn.Parameter(
        layer.weight[:, rank * width : (rank + 1) * width].clone().contiguous(),
        requires_grad=False,
    )
    if layer.bias is not None:
        result.bias = nn.Parameter(layer.bias.detach().clone(), requires_grad=False)
    result.group = group
    return result


def shard_model(model, rank, world_size, group=None):
    """Shard merged CPU weights before moving them to the runner-bound GPU.

    The caller owns the process group. This helper neither reads deployment
    environment variables nor initializes a group or spawns workers.
    """
    if world_size == 1:
        if rank != 0:
            raise ValueError("Single-rank execution requires rank zero")
        return model
    if world_size not in (4, 8) or not 0 <= rank < world_size:
        raise ValueError("Supported tensor-parallel group sizes: 1, 4, 8")
    if not dist.is_initialized():
        raise RuntimeError("The caller must initialize the process group")
    if dist.get_rank(group) != rank or dist.get_world_size(group) != world_size:
        raise ValueError("Tensor-parallel rank/group does not match the process group")
    blocks = [*model.blocks, *model.token_refiner.blocks]
    # Check all shapes before changing any module.
    for block in blocks:
        attn, mlp = block.attn, block.mlp
        if attn.num_heads % world_size:
            raise ValueError("Attention heads must divide across ranks")
        if attn.qkv_proj.out_features != 3 * attn.inner_dim:
            raise ValueError("Expected fused Q/K/V with equal widths")
        if attn.out_proj.in_features != attn.inner_dim:
            raise ValueError("Attention output does not match the complete heads")
        if mlp.fc1.out_features != 2 * mlp.fc2.in_features:
            raise ValueError("Expected fused gate/value feed-forward projection")
        if mlp.fc2.in_features % world_size:
            raise ValueError("MLP width must divide across ranks")
    for block in blocks:
        attn, mlp = block.attn, block.mlp
        attn.qkv_proj = _column_slice(attn.qkv_proj, rank, world_size, 3)
        attn.out_proj = _row_slice(attn.out_proj, rank, world_size, group)
        attn.num_heads //= world_size
        attn.inner_dim //= world_size
        mlp.fc1 = _column_slice(mlp.fc1, rank, world_size, 2)
        mlp.fc2 = _row_slice(mlp.fc2, rank, world_size, group)
    return model
