"""Runtime-independent model contract. All transported values are CPU data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from streaming_cwm_backend import NativeBackend


@dataclass(frozen=True)
class RenderSource:
    """Fresh-run inputs: fitted RGB uint8 (768, 1344, 3), local lossless pairs."""

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
    """Step facts; frames is contiguous RGB uint8 (17, 768, 1344, 3), or None."""

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
        self._backend: NativeBackend | None = None
        self._world_id: int | None = None

    def load(self, *, weights: Path) -> None:
        """Load resolved checkpoints once in the current worker."""
        from streaming_cwm_backend import NativeBackend

        self._backend = NativeBackend(weights)
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
