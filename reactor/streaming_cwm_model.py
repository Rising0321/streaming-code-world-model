"""Runtime-independent model contract. All transported values are CPU data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from streaming_cwm_backend import NativeBackend
    from streaming_cwm_parallel_backend import ParallelBackend


@dataclass(frozen=True)
class RenderSource:
    """Inputs fixed for one render.

    conditions: Directory of validated consecutive depth/semantic pairs.
    image: Fitted opening RGB image, uint8 (768, 1344, 3), on CPU.
    prompt: Static scene description; seed: the fresh run's RNG seed.
    chunks: Number of 17-frame chunks, including the decoder's final flush.
    """

    conditions: Path
    image: np.ndarray
    prompt: str
    seed: int
    chunks: int


@dataclass(frozen=True)
class StreamingCWMInput:
    """Carry the fresh source only until this world_id is acknowledged."""

    world_id: int
    source: RenderSource | None = None


@dataclass(frozen=True)
class StreamingCWMResult:
    """Facts produced by one step.

    world_id: The successfully prepared run identity.
    sampled_chunks: Count of generated chunks, including the pending decode.
    emitted_chunks: Count of decoded 17-frame chunks returned so far.
    complete: True on the final decode-only step.
    frames: Contiguous RGB uint8 (17, 768, 1344, 3) on CPU, or None during lookahead.
    """

    world_id: int
    sampled_chunks: int
    emitted_chunks: int
    complete: bool
    frames: np.ndarray | None = None


class StreamingCWMModel:
    """Own one loaded backend and acknowledge each successfully prepared world.

    An unseen world ID requires a source. Subsequent calls advance one chunk;
    the first produces no frames because decoding needs lookahead. A final
    decode-only call emits the pending 17 frames and reports completion.
    Invalid sequencing and native compute errors propagate to the caller.
    """

    def __init__(self) -> None:
        # DistributedRunner assigns these before load(); defaults allow direct use.
        self.rank, self.world_size, self.device = 0, 1, "cuda:0"
        self._backend: NativeBackend | ParallelBackend | None = None
        self._world_id: int | None = None

    def load(self, *, weights: Path, inference_mode: str = "native") -> None:
        """Load resolved checkpoints once in the current worker."""
        if inference_mode not in ("native", "fa4"):
            raise ValueError("inference_mode must be native or fa4")
        if self.world_size not in (1, 4, 8):
            raise ValueError("Supported GPU counts are 1, 4 and 8")
        if inference_mode == "native":
            if self.world_size != 1:
                raise ValueError("The native reference path uses one GPU")
            from streaming_cwm_backend import NativeBackend

            self._backend = NativeBackend(weights)
        else:
            from streaming_cwm_parallel_backend import ParallelBackend

            self._backend = ParallelBackend(
                weights,
                self.device,
                self.rank,
                self.world_size,
            )
        self._world_id = None

    def generate(self, input: StreamingCWMInput) -> StreamingCWMResult:
        """Advance this input's world; a missing fresh source raises ValueError."""
        if self._backend is None:
            raise RuntimeError("Model has not been loaded")
        if input.world_id != self._world_id:
            if input.source is None:
                raise ValueError("A fresh render requires its source")
            self.reset()
            self._backend.prepare(input.source)
            self._world_id = input.world_id
        frames, sampled, emitted, complete = self._backend.step()
        return StreamingCWMResult(
            world_id=input.world_id,
            sampled_chunks=sampled,
            emitted_chunks=emitted,
            complete=complete,
            frames=frames,
        )

    def reset(self) -> None:
        """Discard the active render while retaining all loaded weights."""
        if self._backend is not None:
            self._backend.reset()
        self._world_id = None

    def shutdown(self) -> None:
        """Release the active render when its worker exits."""
        self.reset()
