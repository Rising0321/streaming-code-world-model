"""CPU/Gloo topology tests; not a substitute for native-checkpoint GPU parity."""

# Optional upstream imports must follow the dependency guards below.

import copy

import pytest

torch = pytest.importorskip("torch")
from reactor_runtime.distributed import DistributedRunner
from streaming_cwm_parallel import shard_model
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim, self.inner_dim = 8, 4, 32
        self.qkv_proj = nn.Linear(16, 96, bias=False, dtype=torch.float64)
        self.out_proj = nn.Linear(32, 16, bias=True, dtype=torch.float64)
        self.cache = []

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = [
            part.reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
            for part in self.qkv_proj(x).split(self.inner_dim, -1)
        ]
        self.cache.append((k, v))
        self.cache = self.cache[-6:]
        keys, values = [torch.cat([pair[i] for pair in self.cache], 2) for i in (0, 1)]
        output = F.scaled_dot_product_attention(q, keys, values)
        return self.out_proj(output.transpose(1, 2).reshape(b, s, self.inner_dim))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 128, bias=True, dtype=torch.float64)
        self.fc2 = nn.Linear(64, 16, bias=True, dtype=torch.float64)

    def forward(self, x):
        gate, value = self.fc1(x).chunk(2, -1)
        return self.fc2(F.silu(gate) * value)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.mlp = Attention(), MLP()

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.token_refiner = nn.Module()
        self.token_refiner.blocks = nn.ModuleList([Block()])

    def forward(self, x):
        for block in [*self.token_refiner.blocks, *self.blocks]:
            x = block(x)
        return x


class ParallelProbe:
    def load(self):
        torch.set_num_threads(1)
        torch.manual_seed(12345)
        self.reference = SmallModel().eval().requires_grad_(False)
        self.parallel = copy.deepcopy(self.reference)
        state = torch.get_rng_state().clone()
        shard_model(self.parallel, self.rank, self.world_size)
        assert torch.equal(state, torch.get_rng_state()), "Weight slicing consumed RNG"

    @torch.inference_mode()
    def generate(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        x = torch.randn(1, 3, 16, generator=generator, dtype=torch.float64)
        expected, actual = self.reference(x), self.parallel(x)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for block in self.parallel.blocks:
            assert block.attn.cache[-1][0].shape[1] == 8 // self.world_size
            assert len(block.attn.cache) == min(index + 1, 6)
        return {"index": index, "max_abs": float((actual - expected).abs().max())}

    def reset(self):
        for model in [self.reference, self.parallel]:
            for block in [*model.blocks, *model.token_refiner.blocks]:
                block.attn.cache.clear()


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="CPU/Gloo suite: set CUDA_VISIBLE_DEVICES=''"
)
@pytest.mark.parametrize("ranks", [1, 4, 8])
def test_forward_cache_window_and_reset(ranks):
    runner = DistributedRunner(
        ParallelProbe, world_size=ranks, start_timeout=90, call_timeout=30
    )
    try:
        runner.start()
        for i in range(12):
            result = runner.generate(i)
            assert result["index"] == i and result["max_abs"] < 1e-12
        runner.reset()
        assert runner.generate(0)["max_abs"] < 1e-12
    finally:
        runner.shutdown()
