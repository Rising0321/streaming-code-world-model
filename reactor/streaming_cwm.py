"""Stream prerecorded depth/semantic conditions through the released CWM."""

import asyncio
import os
from pathlib import Path

from reactor_runtime import (
    ApplicationError,
    ClientInfo,
    CommandError,
    InputField,
    ReactorApp,
    StepOutcome,
    TrackPayload,
    UploadedFile,
    connected,
    event,
    get_weights_path,
    session_ended,
)
from reactor_runtime.distributed import DistributedRunner
from streaming_cwm_images import reference_frames
from streaming_cwm_inputs import StagedSource, example_source
from streaming_cwm_model import (
    RenderSource,
    StreamingCWMInput,
    StreamingCWMModel,
    StreamingCWMResult,
)
from streaming_cwm_types import StateUpdate, StreamingCWMOutput, StreamingCWMState


class StreamingCWM(ReactorApp):
    """An explicit Start accepts the complete input; generation then advances by chunk."""

    fps = 24
    state: StreamingCWMState

    def load(self, config_path: Path | None) -> None:
        from streaming_cwm_assets import (
            download_weights,
            load_settings,
            prepare_demo_inputs,
        )

        self._num_gpus, self._inference_mode = load_settings(config_path)
        self._weights = Path(
            os.environ.get("STREAMING_CWM_WEIGHTS", str(get_weights_path()))
        )
        download_weights(self._weights)
        self._examples = prepare_demo_inputs(
            self._weights,
            Path(os.environ.get("STREAMING_CWM_CWM_SOURCE", "/opt/cwm")),
        )
        self._input_root = self._weights / "uploads"
        self._start_engine()

    def _start_engine(self):
        self._engine = DistributedRunner(
            StreamingCWMModel,
            world_size=self._num_gpus,
            load_kwargs={
                "weights": self._weights,
                "inference_mode": self._inference_mode,
            },
            start_timeout=1200.0,
            call_timeout=900.0,
        )
        self._engine.start()

    def _status(self) -> StateUpdate:
        s = self.state
        return StateUpdate(
            world_id=s._world_id,
            active=s._active,
            ready=s._applied_world_id == s._world_id,
            sampled_chunks=s._sampled,
            completed_chunks=s._emitted,
            total_chunks=s._total,
            output_frames=17 * s._emitted,
            unused_tail_frames=s._unused,
            complete=s._complete,
            last_chunk_seconds=s._seconds,
        )

    @event(
        name="start",
        description=(
            "Start a fresh render from prerecorded lossless depth/semantic pairs, an aligned "
            "reference image and a static prompt. Valid at any time; all inputs are validated "
            "before replacing a run. Refuses invalid input with invalid_input and keeps "
            "the previous render intact. Broadcasts state_update. No live keyboard controls."
        ),
    )
    async def start(
        self,
        conditions: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "ZIP up to 512 MiB containing consecutive 000000.depth.f32 and "
                "000000.semantic_id.png pairs at 336 x 192. Depth: little-endian float32 "
                "metres. Semantic: grayscale IDs 0–11. At least 102 frames; no MP4 input."
            ),
        ),
        image: UploadedFile = InputField(  # noqa: B008 - schema field declaration
            moderate=True,
            description=(
                "First-frame appearance reference aligned with the condition sequence; "
                "PNG, JPEG or WebP up to 16 MiB. Resized to 1344 x 768 as in native inference."
            ),
        ),
        prompt: str = InputField(
            moderate=True,
            min_length=1,
            max_length=32768,
            description=(
                "Static appearance and scene description. Fixed during a render. Native "
                "preparation removes the Protagonist action timeline section, when present."
            ),
        ),
        seed: int = InputField(
            default=0,
            ge=0,
            le=2147483647,
            description="Random seed for this fresh render.",
        ),
        max_chunks: int = InputField(
            default=24,
            ge=0,
            le=480,
            description=(
                "Maximum 17-frame chunks. Zero uses all complete chunks, up to 480. "
                "At least six must be available and selected; trailing frames are reported."
            ),
        ),
    ) -> StateUpdate:
        try:
            staged = await asyncio.to_thread(
                StagedSource,
                conditions.data,
                image.data,
                prompt,
                seed,
                max_chunks,
                self._input_root,
            )
        except ValueError as error:
            raise CommandError("invalid_input", str(error)) from error
        return await self._queue_source(staged.source, staged.unused_frames, staged)

    @event(
        name="start_example",
        description=(
            "Render one installed official example with its paired reference image. "
            "Conditions are read on the server. Inputs stay fixed until completion or Stop. "
            "Broadcasts state_update. Refuses invalid inputs with invalid_example and "
            "leaves the previous render intact."
        ),
    )
    async def start_example(
        self,
        example_id: int = InputField(
            ge=1, le=40, description="Official CWM example number, from 1 to 40."
        ),
        prompt: str = InputField(
            moderate=True,
            min_length=1,
            max_length=32768,
            description="Static appearance and scene prompt for this render; action timeline text is removed.",
        ),
        seed: int = InputField(
            default=0,
            ge=0,
            le=2147483647,
            description="Random seed for the fresh render.",
        ),
        max_chunks: int = InputField(
            default=24,
            ge=0,
            le=480,
            description="17-frame chunks to render: 6–480, or zero for all complete chunks.",
        ),
    ) -> StateUpdate:
        try:
            source, unused = await asyncio.to_thread(
                example_source, self._examples, example_id, prompt, seed, max_chunks
            )
        except ValueError as error:
            raise CommandError("invalid_example", str(error)) from error
        return await self._queue_source(source, unused)

    async def _queue_source(
        self, source: RenderSource, unused: int, staged: StagedSource | None = None
    ) -> StateUpdate:
        self._release_input()
        self.state._staged = staged
        s = self.state
        s._world_id += 1
        s._source = source
        s._reference_conditions = source.conditions
        s._active, s._complete = True, False
        s._sampled = s._emitted = 0
        s._total, s._unused = source.chunks, unused
        s._seconds = None
        self.output.flush()
        status = self._status()
        await self.send(status)
        return status

    @event(
        name="stop",
        description=(
            "Stop rendering and clear queued video. Safe to call while idle. "
            "Start again to render a fresh video. Broadcasts state_update."
        ),
    )
    async def stop_render(self) -> StateUpdate:
        self._engine.reset()
        self._release_input()
        self.state._active = False
        self.state._source = None
        self.state._applied_world_id = None
        self.output.flush()
        status = self._status()
        await self.send(status)
        return status

    async def process_input(self) -> StreamingCWMInput:
        s = self.state
        if not s._active:
            raise ApplicationError("Select inputs and press Start rendering")
        return StreamingCWMInput(
            world_id=s._world_id,
            source=s._source if s._applied_world_id != s._world_id else None,
        )

    def generate(self, input: StreamingCWMInput) -> StreamingCWMResult:
        return self._engine.generate(input)

    async def process_output(self, outcome: StepOutcome) -> StreamingCWMOutput | None:
        if outcome.error is not None:
            # A native error can leave partially advanced caches. Let Runtime
            # end the session; never acknowledge or count a failed step.
            raise outcome.error
        result: StreamingCWMResult = outcome.result
        s = self.state
        if result.world_id != s._world_id:
            raise ValueError("Result belongs to a different rendering run")
        reference = None
        if result.frames is not None:
            if (
                s._reference_conditions is None
                or result.emitted_chunks != s._emitted + 1
            ):
                raise ValueError(
                    "Generated chunk does not match its reference sequence"
                )
            if result.frames.dtype != "uint8" or result.frames.shape != (
                17,
                768,
                1344,
                3,
            ):
                raise ValueError("Expected one native 17-frame RGB chunk")
            reference = await asyncio.to_thread(
                reference_frames, s._reference_conditions, result.emitted_chunks - 1
            )
        s._applied_world_id = result.world_id
        s._source = None
        s._sampled, s._emitted = result.sampled_chunks, result.emitted_chunks
        s._complete, s._active = result.complete, not result.complete
        s._seconds = outcome.elapsed
        # Keep uploaded semantic frames until the final reference has been read.
        if result.complete:
            self._release_input()
        await self.send(self._status())
        if reference is None:
            return None
        metadata = [
            {
                "world_id": result.world_id,
                "chunk": result.emitted_chunks - 1,
                "frame": 17 * (result.emitted_chunks - 1) + index,
            }
            for index in range(17)
        ]
        return StreamingCWMOutput(
            main_video=TrackPayload(result.frames, metadata=metadata),
            reference_video=TrackPayload(reference, metadata=metadata),
        )

    def _release_input(self):
        self.state._reference_conditions = None
        if self.state._staged is not None:
            self.state._staged.close()
            self.state._staged = None

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        await client.send(self._status())

    @session_ended
    def on_session_ended(self) -> None:
        self._release_input()
        self.output.flush()
        if self._engine.healthy:
            self._engine.reset()
        else:
            self._engine.shutdown()
            self._start_engine()
