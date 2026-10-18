"""Runtime lifecycle and lossless upload contract, without weights or a GPU."""

import asyncio
import io
import subprocess
import sys
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from reactor_runtime import ApplicationError, CommandError, StepOutcome, UploadedFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streaming_cwm import StreamingCWM
from streaming_cwm_inputs import StagedSource, example_source, static_prompt
from streaming_cwm_model import (
    RenderSource,
    StreamingCWMInput,
    StreamingCWMModel,
    StreamingCWMResult,
)
from streaming_cwm_types import StreamingCWMState


def png(array, mode):
    out = io.BytesIO()
    Image.fromarray(array, mode).save(out, format="PNG")
    return out.getvalue()


IMAGE = png(np.zeros((8, 8, 3), np.uint8), "RGB")
SEMANTIC = png(np.ones((192, 336), np.uint8), "L")


def archive(frames=102, bad=None):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(frames):
            z.writestr(f"{i:06d}.depth.f32", np.ones((192, 336), dtype="<f4").tobytes())
            z.writestr(f"{i:06d}.semantic_id.png", SEMANTIC)
        if bad:
            z.writestr(bad, b"bad")
    return out.getvalue()


class FakeEngine:
    healthy = True

    def __init__(self):
        self.inputs, self.resets = [], 0

    def generate(self, value):
        self.inputs.append(value)
        return StreamingCWMResult(
            world_id=value.world_id, sampled_chunks=1, emitted_chunks=0, complete=False
        )

    def reset(self):
        self.resets += 1


def app(tmp_path):
    value = StreamingCWM()
    value.state = StreamingCWMState()
    value._engine = FakeEngine()
    value._input_root = tmp_path
    messages = []

    async def send(message):
        messages.append(message)

    value.send = send
    return value


def start(value, *, data=None, prompt="A sunny road.", limit=6, seed=0):
    return asyncio.run(
        value.start(
            conditions=UploadedFile(
                name="conditions.zip",
                mime_type="application/zip",
                data=archive() if data is None else data,
            ),
            image=UploadedFile(name="reference.png", mime_type="image/png", data=IMAGE),
            prompt=prompt,
            seed=seed,
            max_chunks=limit,
        )
    )


def step_input(value):
    return asyncio.run(value.process_input())


def test_refused_step_never_reaches_model(tmp_path):
    value = app(tmp_path)
    with pytest.raises(ApplicationError):
        value.generate(step_input(value))
    assert not value._engine.inputs


def test_anchor_until_ack_and_forward_only(tmp_path):
    value = app(tmp_path)
    start(value)
    snapshot = step_input(value)
    assert snapshot.source is not None and step_input(value).source is snapshot.source
    with pytest.raises(FrozenInstanceError):
        snapshot.world_id = 9
    value.state._world_id = 99
    result = value.generate(snapshot)
    assert value._engine.inputs[-1] is snapshot and result.world_id == 1
    value.state._world_id = 1
    asyncio.run(value.process_output(StepOutcome(result=result, elapsed=0.25)))
    assert step_input(value).source is None
    assert value.state._seconds == 0.25 and value.state._sampled == 1
    assert not list(tmp_path.iterdir())


def test_error_does_not_ack_or_complete(tmp_path):
    value = app(tmp_path)
    start(value)
    with pytest.raises(RuntimeError, match="native failure"):
        asyncio.run(
            value.process_output(
                StepOutcome(error=RuntimeError("native failure"), elapsed=0.5)
            )
        )
    assert value.state._sampled == value.state._emitted == 0
    assert step_input(value).source is not None
    value.on_session_ended()
    assert value._engine.resets == 1 and not list(tmp_path.iterdir())


def test_invalid_upload_keeps_active_render(tmp_path):
    value = app(tmp_path)
    start(value, seed=42)
    original = step_input(value)
    for data, prompt, limit in [
        (b"no", "valid", 6),
        (archive(), " ", 6),
        (archive(), "valid", 1),
        (archive(bad="../escape"), "valid", 6),
    ]:
        with pytest.raises(CommandError):
            start(value, data=data, prompt=prompt, limit=limit)
        assert step_input(value) == original
        assert len(list(tmp_path.iterdir())) == 1
    start(value, seed=0)
    assert step_input(value).world_id == 2
    assert step_input(value).source.seed == 0
    assert len(list(tmp_path.iterdir())) == 1
    asyncio.run(value.stop_render())
    assert not list(tmp_path.iterdir())


def test_final_output_precedes_refusal(tmp_path):
    value = app(tmp_path)
    start(value)
    frames = np.zeros((17, 8, 8, 3), np.uint8)
    result = StreamingCWMResult(
        world_id=1, sampled_chunks=6, emitted_chunks=6, complete=True, frames=frames
    )
    output = asyncio.run(value.process_output(StepOutcome(result=result, elapsed=0.1)))
    assert output is not None
    assert value.state._complete and not value.state._active
    assert value._status().output_frames == 102
    with pytest.raises(ApplicationError):
        step_input(value)


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",
        "/absolute",
        "conditions/../../escape",
        "000000.depth.f32",
        "video.mp4",
    ],
)
def test_unsafe_zip_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        StagedSource(archive(bad=bad), IMAGE, "scene", 0, 6, tmp_path)
    assert not list(tmp_path.iterdir())


def test_native_static_caption_rule():
    assert (
        static_prompt(
            "Preamble\nAppearance and scene: road\nProtagonist action timeline: turn"
        )
        == "Appearance and scene: road"
    )
    with pytest.raises(ValueError):
        static_prompt("Protagonist action timeline: turn")


def test_model_half_owns_new_world_and_errors():
    class Backend:
        def reset(self):
            self.count = 0

        def prepare(self, source):
            self.source = source

        def step(self):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("failed")
            return None, self.count, 0, False

    model = StreamingCWMModel()
    model._backend, model._world_id = Backend(), None
    with pytest.raises(ValueError):
        model.generate(StreamingCWMInput(world_id=1))
    source = RenderSource(
        conditions=Path("conditions"),
        image=np.zeros((768, 1344, 3), np.uint8),
        prompt="road",
        seed=0,
        chunks=6,
    )
    assert (
        model.generate(StreamingCWMInput(world_id=1, source=source)).sampled_chunks == 1
    )
    with pytest.raises(RuntimeError, match="failed"):
        model.generate(StreamingCWMInput(world_id=1))
    model.reset()
    assert model._world_id is None
    assert (
        model.generate(StreamingCWMInput(world_id=2, source=source)).sampled_chunks == 1
    )


def test_model_contract_imports_without_runtime():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("reactor_runtime"):
            raise ImportError("Runtime is forbidden in the model half")
sys.meta_path.insert(0, Block())
import streaming_cwm_model
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def test_new_session_has_no_previous_inputs_or_seed(tmp_path):
    value = app(tmp_path)
    start(value, seed=42)
    value.on_session_ended()
    value.state = StreamingCWMState()
    assert value.state._source is None and value.state._world_id == 0
    assert not value.state._active and not list(tmp_path.iterdir())


def test_official_example_and_invalid_replacement(tmp_path):
    value = app(tmp_path / "uploads")
    value._examples = tmp_path / "examples"
    directory = value._examples / "example_02"
    conditions = directory / "conditions"
    conditions.mkdir(parents=True)
    (directory / "first_frame.png").write_bytes(IMAGE)
    for i in range(110):
        (conditions / f"{i:06d}.semantic_id.png").write_bytes(SEMANTIC)
    status = asyncio.run(
        value.start_example(example_id=2, prompt="A quiet road", seed=42, max_chunks=6)
    )
    assert status.world_id == 1 and status.unused_tail_frames == 8
    source = step_input(value).source
    assert source.conditions == conditions and source.seed == 42
    assert source.image.shape == (768, 1344, 3)
    assert value.state._staged is None
    with pytest.raises(CommandError):
        asyncio.run(
            value.start_example(example_id=41, prompt="road", seed=0, max_chunks=6)
        )
    assert step_input(value).source is source
    value.generate(step_input(value))
    assert len(value._engine.inputs) == 1
    asyncio.run(value.stop_render())
    assert conditions.exists()  # Shared downloaded examples never belong to a session.


@pytest.mark.parametrize(
    "example,seed,limit", [(0, 0, 6), (41, 0, 6), (1, -1, 6), (1, 0, 481)]
)
def test_invalid_example_arguments(tmp_path, example, seed, limit):
    with pytest.raises(ValueError):
        example_source(tmp_path, example, "scene", seed, limit)


def test_runtime_drives_real_async_input_hook(tmp_path):
    """Use the installed Runtime loop, including its await contract and refusal gate."""
    value = app(tmp_path)

    async def exercise():
        value._on_loop_ready()
        value._live.set()
        reported = asyncio.Event()
        value._report_step = lambda *args: reported.set()
        task = asyncio.create_task(value.run())
        try:
            await asyncio.sleep(0.03)
            assert not task.done() and not value._engine.inputs
            source = RenderSource(
                conditions=tmp_path,
                image=np.zeros((768, 1344, 3), np.uint8),
                prompt="scene",
                seed=0,
                chunks=6,
            )
            await value._queue_source(source, 0)
            await asyncio.wait_for(reported.wait(), timeout=1)
            assert value._engine.inputs[0].source is source
            assert value.state._applied_world_id == 1
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    asyncio.run(exercise())
