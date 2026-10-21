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
from streaming_cwm_images import PALETTE, reference_frames
from streaming_cwm_inputs import (
    StagedSource,
    decode_reference,
    example_source,
    static_prompt,
)
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


def archive(frames=102, bad=None, depth=None, semantic=SEMANTIC):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(frames):
            z.writestr(
                f"{i:06d}.depth.f32",
                np.ones((192, 336), dtype="<f4").tobytes() if depth is None else depth,
            )
            z.writestr(f"{i:06d}.semantic_id.png", semantic)
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
    assert len(list(tmp_path.iterdir())) == 1


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
    value.state._emitted = 5
    frames = np.zeros((17, 768, 1344, 3), np.uint8)
    result = StreamingCWMResult(
        world_id=1, sampled_chunks=6, emitted_chunks=6, complete=True, frames=frames
    )
    output = asyncio.run(value.process_output(StepOutcome(result=result, elapsed=0.1)))
    assert output is not None
    assert value.state._complete and not value.state._active
    assert value._status().output_frames == 102
    assert not list(tmp_path.iterdir())
    with pytest.raises(ApplicationError):
        step_input(value)


def test_reference_uses_emitted_frame_indices(tmp_path):
    # Distinct IDs at the chunk boundary catch a one-chunk lookahead offset.
    for index in range(34):
        (tmp_path / f"{index:06d}.semantic_id.png").write_bytes(
            png(np.full((192, 336), index % 12, np.uint8), "L")
        )
    reference = reference_frames(tmp_path, 1)
    assert reference.shape == (17, 192, 336, 3)
    for offset in range(17):
        np.testing.assert_array_equal(
            reference[offset],
            np.broadcast_to(PALETTE[(17 + offset) % 12], (192, 336, 3)),
        )


def test_reference_survives_lookahead_and_all_chunks_until_final_flush(tmp_path):
    value = app(tmp_path)
    start(value)
    reference = value.state._reference_conditions
    asyncio.run(
        value.process_output(
            StepOutcome(
                result=StreamingCWMResult(
                    world_id=1, sampled_chunks=1, emitted_chunks=0, complete=False
                ),
                elapsed=0.1,
            )
        )
    )
    assert reference.is_dir() and step_input(value).source is None
    frames = np.zeros((17, 768, 1344, 3), np.uint8)
    for emitted in range(1, 7):
        result = StreamingCWMResult(
            world_id=1,
            sampled_chunks=min(emitted + 1, 6),
            emitted_chunks=emitted,
            complete=emitted == 6,
            frames=frames,
        )
        output = asyncio.run(
            value.process_output(StepOutcome(result=result, elapsed=0.1))
        )
        assert output.main_video is frames
        assert output.reference_video.shape == (17, 192, 336, 3)
        assert (
            output.__metadata__["main_video"] == output.__metadata__["reference_video"]
        )
        assert output.__metadata__["main_video"][0]["frame"] == (emitted - 1) * 17
        assert output.__metadata__["main_video"][-1]["frame"] == emitted * 17 - 1
        assert reference.is_dir() == (emitted != 6)
    assert value.state._reference_conditions is None


def test_mismatched_output_cannot_advance_reference_or_progress(tmp_path):
    value = app(tmp_path)
    start(value)
    for world_id, emitted in [(99, 1), (1, 2)]:
        with pytest.raises(ValueError):
            asyncio.run(
                value.process_output(
                    StepOutcome(
                        result=StreamingCWMResult(
                            world_id=world_id,
                            sampled_chunks=2,
                            emitted_chunks=emitted,
                            complete=False,
                            frames=np.zeros((17, 768, 1344, 3), np.uint8),
                        ),
                        elapsed=0.1,
                    )
                )
            )
        assert value.state._emitted == value.state._sampled == 0
        assert value.state._reference_conditions.is_dir()
    value.on_session_ended()
    assert not list(tmp_path.iterdir())


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


@pytest.mark.parametrize("depth", [float("nan"), float("inf"), -1.0])
def test_invalid_depth_rejected(tmp_path, depth):
    data = np.full((192, 336), depth, dtype="<f4").tobytes()
    with pytest.raises(ValueError, match="distances"):
        StagedSource(archive(depth=data), IMAGE, "scene", 0, 6, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "semantic", [b"not a png", IMAGE, png(np.full((192, 336), 12, np.uint8), "L")]
)
def test_invalid_semantic_rejected(tmp_path, semantic):
    with pytest.raises(ValueError, match="Semantic"):
        StagedSource(archive(semantic=semantic), IMAGE, "scene", 0, 6, tmp_path)
    assert not list(tmp_path.iterdir())


def test_upload_size_limits(tmp_path, monkeypatch):
    import streaming_cwm_inputs as inputs

    monkeypatch.setattr(inputs, "MAX_UPLOAD", 4)
    with pytest.raises(ValueError, match="512 MiB"):
        StagedSource(b"12345", IMAGE, "scene", 0, 6, tmp_path)
    with pytest.raises(ValueError, match="16 MiB"):
        decode_reference(b"x" * (16 * 1024**2 + 1))
    monkeypatch.setattr(inputs, "MAX_UPLOAD", 512 * 1024**2)
    monkeypatch.setattr(inputs, "MAX_EXPANDED", 1)
    with pytest.raises(ValueError, match="3 GiB"):
        StagedSource(archive(), IMAGE, "scene", 0, 6, tmp_path)


def test_disk_error_is_not_an_invalid_upload(tmp_path, monkeypatch):
    def fail_write(path, data):
        raise OSError("test storage failure")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(OSError, match="storage failure"):
        StagedSource(archive(), IMAGE, "scene", 0, 6, tmp_path)
    assert not list(tmp_path.iterdir())


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


@pytest.mark.parametrize(
    "count,mode", [(1, "native"), (1, "fa4"), (4, "fa4"), (8, "fa4")]
)
def test_resolved_deployment_settings(tmp_path, count, mode):
    from streaming_cwm_assets import load_settings

    config = tmp_path / "settings.yaml"
    config.write_text(f"num_gpus: {count}\ninference_mode: {mode}\n")
    assert load_settings(config) == (count, mode)


@pytest.mark.parametrize(
    "content",
    [
        "num_gpus: true\ninference_mode: fa4",
        "num_gpus: 2\ninference_mode: fa4",
        "num_gpus: 4\ninference_mode: native",
        "num_gpus: 1\ninference_mode: fp8",
        "[]",
        "num_gpus: 1\ninference_mode: native\nunknown: true",
    ],
)
def test_invalid_deployment_fails_before_workers(tmp_path, content):
    from streaming_cwm_assets import load_settings

    config = tmp_path / "settings.yaml"
    config.write_text(content)
    with pytest.raises(ValueError):
        load_settings(config)


def test_runner_receives_resolved_settings_and_recovers(tmp_path, monkeypatch):
    import streaming_cwm

    runners = []

    class Runner:
        healthy = False

        def __init__(self, cls, **kwargs):
            assert cls is StreamingCWMModel
            self.settings, self.started, self.stopped = kwargs, False, False
            runners.append(self)

        def start(self):
            self.started = True

        def shutdown(self):
            self.stopped = True

    monkeypatch.setattr(streaming_cwm, "DistributedRunner", Runner)
    value = app(tmp_path)
    value._weights, value._num_gpus, value._inference_mode = tmp_path, 8, "fa4"
    value._start_engine()
    assert runners[0].started and runners[0].settings["world_size"] == 8
    assert runners[0].settings["load_kwargs"] == {
        "weights": tmp_path,
        "inference_mode": "fa4",
    }
    value.on_session_ended()
    assert runners[0].stopped and runners[1].started


def test_refusal_is_private_and_internal_errors_are_not_client_input_errors(
    tmp_path, monkeypatch
):
    import streaming_cwm

    value = app(tmp_path)
    messages = []

    async def send(message):
        messages.append(message)

    value.send = send
    with pytest.raises(CommandError) as error:
        start(value, data=b"bad")
    assert error.value.code == "invalid_input" and not messages

    def unavailable(*args):
        raise OSError("internal filesystem failure")

    monkeypatch.setattr(streaming_cwm, "StagedSource", unavailable)
    with pytest.raises(OSError):
        start(value)
    assert not messages


def test_model_dispatches_native_and_fa4_without_deployment_reads(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "streaming_cwm_backend",
        SimpleNamespace(
            NativeBackend=lambda weights: calls.append(("native", weights)) or object()
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "streaming_cwm_parallel_backend",
        SimpleNamespace(
            ParallelBackend=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or object()
            )
        ),
    )
    model = StreamingCWMModel()
    model.load(weights=tmp_path)
    assert calls[-1] == ("native", tmp_path)
    model.rank, model.world_size, model.device = 3, 8, "cuda:3"
    model.load(weights=tmp_path, inference_mode="fa4")
    assert calls[-1] == ((tmp_path, "cuda:3", 3, 8), {})
    with pytest.raises(ValueError):
        model.load(weights=tmp_path, inference_mode="native")
