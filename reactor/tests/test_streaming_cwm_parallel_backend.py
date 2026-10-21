"""CPU bookkeeping tests for the multi-rank service backend."""

# Optional upstream imports must follow the dependency guards below.

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rtcwm")
pytest.importorskip("musubi_tuner")

import streaming_cwm_parallel_backend as live_model
from streaming_cwm_model import StreamingCWMModel


class FakeSampler:
    def __init__(self, model, window_sample, chunks, seed, device):
        self.chunks, self.seed, self.index = chunks, seed, 0

    def step(self):
        self.index += 1
        return self.index


class FakeDecoder:
    def __init__(self, vae):
        self.pending = None

    def push(self, latent):
        previous, self.pending = self.pending, latent
        return None if previous is None else np.full((17, 2, 2, 3), previous, np.uint8)

    def finish(self):
        return np.full((17, 2, 2, 3), self.pending, np.uint8)


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_leader_prepares_and_all_ranks_keep_decode_cadence(monkeypatch, rank):
    monkeypatch.setattr(live_model, "ChunkSampler", FakeSampler)

    class FollowerDecoder(FakeDecoder):
        def push(self, latent):
            self.pending = latent

        def finish(self):
            assert self.pending == 12

    import streaming_cwm_infra as infra

    monkeypatch.setattr(
        infra, "GPUBytesDecoder", FakeDecoder if rank == 0 else FollowerDecoder
    )
    prepared = SimpleNamespace(window_sample=lambda index: {})
    local_prepares, broadcasts = [], []

    def broadcast(values, src, device):
        broadcasts.append(src)
        if rank != 0:
            assert values == [None, None]
        values[0] = prepared

    monkeypatch.setattr(live_model.dist, "broadcast_object_list", broadcast)
    backend = live_model.ParallelBackend.__new__(live_model.ParallelBackend)
    backend.rank, backend.world_size, backend.device = rank, 8, torch.device("cpu")
    backend.model, backend.vae = object(), object()
    backend.packed_kv = SimpleNamespace(chunk=nullcontext)
    backend._prepare_on_leader = lambda source: (
        local_prepares.append(source) or prepared
    )
    source = SimpleNamespace(chunks=12, seed=42)
    backend.prepare(source)
    assert len(local_prepares) == (1 if rank == 0 else 0)
    assert broadcasts == [0] and backend._run.sampler.seed == 42
    assert backend._run.decoder is not None
    outputs = [backend.step() for _ in range(13)]
    assert [o[1] for o in outputs] == [*range(1, 13), 12]
    assert [o[3] for o in outputs] == [False] * 12 + [True]
    if rank == 0:
        assert outputs[0][0] is None
        assert [o[2] for o in outputs] == list(range(13))
        assert sum(len(o[0]) for o in outputs if o[0] is not None) == 204
    else:
        assert all(o[0] is None for o in outputs)
    assert backend._run is None
    with pytest.raises(RuntimeError, match="No active render"):
        backend.step()


def test_preparation_error_is_not_acknowledged():
    model = StreamingCWMModel()

    class Backend:
        def reset(self):
            pass

        def prepare(self, source):
            raise ValueError("encoder failed")

    model._backend = Backend()
    with pytest.raises(ValueError, match="encoder failed"):
        model.generate(SimpleNamespace(world_id=1, source=object()))
    assert model._world_id is None


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_prepare_failure_reaches_every_rank_before_sampling(monkeypatch, rank):
    payloads = []

    def failed(source):
        raise ValueError("encoder failed")

    def broadcast(payload, src, device):
        payloads.append(list(payload))
        payload[1] = "ValueError: encoder failed"

    monkeypatch.setattr(live_model.dist, "broadcast_object_list", broadcast)
    backend = live_model.ParallelBackend.__new__(live_model.ParallelBackend)
    backend.rank, backend.world_size, backend.device = rank, 8, torch.device("cpu")
    backend._prepare_on_leader = failed
    with pytest.raises(RuntimeError, match="Input preparation failed"):
        backend.prepare(SimpleNamespace(chunks=12, seed=0))
    assert len(payloads) == 1 and backend._run is None
    assert (payloads[0][1] is not None) == (rank == 0)
