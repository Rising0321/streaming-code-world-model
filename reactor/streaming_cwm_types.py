"""Public progress messages and private per-session input state."""

from pathlib import Path

from reactor_runtime import InputState, MessageField, ModelMessage, Output, Video
from streaming_cwm_inputs import StagedSource
from streaming_cwm_model import RenderSource


class StreamingCWMState(InputState):
    _staged: StagedSource | None = None
    _world_id: int = 0
    _applied_world_id: int | None = None
    _source: RenderSource | None = None
    _reference_conditions: Path | None = None
    _active: bool = False
    _complete: bool = False
    _sampled: int = 0
    _emitted: int = 0
    _total: int = 0
    _unused: int = 0
    _seconds: float | None = None


class StreamingCWMOutput(Output):
    main_video: Video
    reference_video: Video


class StateUpdate(ModelMessage):
    world_id: int = MessageField(description="Identity of this rendering run.")
    active: bool = MessageField(
        description="Whether more video remains to be generated or published."
    )
    ready: bool = MessageField(
        description="Whether this run has finished preparing its inputs."
    )
    sampled_chunks: int = MessageField(
        description="Successfully generated chunks, including any waiting to be decoded."
    )
    completed_chunks: int = MessageField(
        description="Decoded 17-frame chunks published on main_video."
    )
    total_chunks: int = MessageField(description="Number of 17-frame chunks requested.")
    output_frames: int = MessageField(
        description="Total frames published to the video track."
    )
    unused_tail_frames: int = MessageField(
        description="Source frames excluded by the limit or incomplete final chunk."
    )
    complete: bool = MessageField(
        description="Final chunk published; buffered video may still be playing."
    )
    last_chunk_seconds: float | None = MessageField(
        description="Runtime-measured duration of the last step, including decode."
    )
