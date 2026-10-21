"""Validate and stage lossless input pairs before changing an active render."""

import io
import re
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from streaming_cwm_model import RenderSource

MAX_CHUNKS = 480
MAX_UPLOAD = 512 * 1024**2
MAX_EXPANDED = 3 * 1024**3
FRAME_BYTES = 192 * 336 * 4
NAME = re.compile(r"(?:conditions/)?(\d{6})\.(depth\.f32|semantic_id\.png)\Z")


def static_prompt(value: str) -> str:
    """The native open_world.static_caption rule, without importing GPU modules."""
    head = "Appearance and scene:"
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = head + value.split(head, 1)[1] if head in value else value
    value = value.split("Protagonist action timeline:")[0].strip()
    if not value:
        raise ValueError("Provide a static appearance/scene prompt")
    return value


def decode_reference(image: bytes) -> np.ndarray:
    if not image or len(image) > 16 * 1024**2:
        raise ValueError("Reference image must be non-empty and at most 16 MiB")
    try:
        with Image.open(io.BytesIO(image)) as im:
            if (
                im.format not in ("PNG", "JPEG", "WEBP")
                or im.width * im.height > 40_000_000
            ):
                raise ValueError(
                    "Use a PNG, JPEG or WebP reference up to 40 million pixels"
                )
            # Match native _load_anchor: RGB, fixed resolution, LANCZOS.
            return np.array(
                im.convert("RGB").resize((1344, 768), Image.Resampling.LANCZOS),
                dtype=np.uint8,
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Reference image cannot be decoded") from error


def example_source(
    root: Path, example_id: int, prompt: str, seed: int, max_chunks: int
) -> tuple[RenderSource, int]:
    """Select an installed, trusted official input; never resolve a client path."""
    if (
        not 1 <= example_id <= 40
        or not 0 <= seed <= 2147483647
        or not 0 <= max_chunks <= MAX_CHUNKS
    ):
        raise ValueError("Invalid example, seed or chunk limit")
    directory = root / f"example_{example_id:02d}"
    conditions = directory / "conditions"
    frames = len(list(conditions.glob("*.semantic_id.png")))
    chunks = min(frames // 17, max_chunks or MAX_CHUNKS)
    if chunks < 6:
        raise ValueError("Select at least six chunks from an installed example")
    return RenderSource(
        conditions=conditions,
        image=decode_reference((directory / "first_frame.png").read_bytes()),
        prompt=static_prompt(prompt),
        seed=seed,
        chunks=chunks,
    ), frames - chunks * 17


class StagedSource:
    """Own only the unique temporary directory created for this accepted upload."""

    def __init__(
        self,
        archive: bytes,
        image: bytes,
        prompt: str,
        seed: int,
        max_chunks: int,
        root: Path,
    ):
        self._temp = None
        prompt = static_prompt(prompt)
        if not (0 <= seed <= 2147483647 and 0 <= max_chunks <= MAX_CHUNKS):
            raise ValueError("Invalid seed or chunk limit")
        if not archive or len(archive) > MAX_UPLOAD:
            raise ValueError("Conditions ZIP must be non-empty and at most 512 MiB")
        reference = decode_reference(image)
        root.mkdir(parents=True, exist_ok=True)
        self._temp = tempfile.TemporaryDirectory(prefix="render-", dir=root)
        directory = Path(self._temp.name)
        try:
            conditions = directory / "conditions"
            conditions.mkdir()
            try:
                with zipfile.ZipFile(io.BytesIO(archive)) as z:
                    entries = [i for i in z.infolist() if not i.is_dir()]
                    if len(entries) > MAX_CHUNKS * 17 * 2 + 32:
                        raise ValueError("Too many condition files")
                    if sum(i.file_size for i in entries) > MAX_EXPANDED:
                        raise ValueError("Expanded conditions exceed 3 GiB")
                    found = {}
                    for entry in entries:
                        if (
                            entry.compress_type
                            not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                            or entry.flag_bits & 1
                        ):
                            raise ValueError(
                                "Use an unencrypted ZIP with store or deflate compression"
                            )
                        match = NAME.fullmatch(entry.filename)
                        if match is None or stat.S_ISLNK(entry.external_attr >> 16):
                            raise ValueError(
                                "ZIP must contain only NNNNNN.depth.f32 / NNNNNN.semantic_id.png pairs"
                            )
                        index, kind = int(match[1]), match[2]
                        if (index, kind) in found:
                            raise ValueError("Duplicate condition frame")
                        if entry.file_size > (
                            FRAME_BYTES if kind == "depth.f32" else 1024**2
                        ):
                            raise ValueError("Oversized condition frame")
                        found[index, kind] = entry
                    total = len(found) // 2
                    expected = {
                        (i, k)
                        for i in range(total)
                        for k in ("depth.f32", "semantic_id.png")
                    }
                    if set(found) != expected:
                        raise ValueError(
                            "Depth and semantic frames must be paired and consecutive from 000000"
                        )
                    chunks = min(total // 17, max_chunks or MAX_CHUNKS)
                    if chunks < 6:
                        raise ValueError(
                            "Provide at least 102 frames (six native chunks)"
                        )
                    if total // 17 > MAX_CHUNKS:
                        raise ValueError("Conditions exceed 480 chunks")
                    for (index, kind), entry in found.items():
                        data = z.read(entry)
                        if kind == "depth.f32":
                            if len(data) != FRAME_BYTES:
                                raise ValueError(
                                    "Depth must be little-endian float32, 336 x 192"
                                )
                            depth = np.frombuffer(data, dtype="<f4")
                            if not np.isfinite(depth).all() or (depth < 0).any():
                                raise ValueError(
                                    "Depth contains non-finite or negative distances"
                                )
                        else:
                            try:
                                with Image.open(io.BytesIO(data)) as im:
                                    if (
                                        im.format != "PNG"
                                        or im.mode != "L"
                                        or im.size != (336, 192)
                                    ):
                                        raise ValueError(
                                            "Semantic ID must be a 336 x 192 grayscale PNG"
                                        )
                                    if np.asarray(im).max() > 11:
                                        raise ValueError(
                                            "Semantic IDs must be in [0, 11]"
                                        )
                            except (OSError, Image.DecompressionBombError) as error:
                                raise ValueError(
                                    "Semantic PNG cannot be decoded"
                                ) from error
                        if index < chunks * 17:
                            (conditions / f"{index:06d}.{kind}").write_bytes(data)
            except (zipfile.BadZipFile, zlib.error) as error:
                raise ValueError("Conditions ZIP cannot be decoded") from error
            self.total_frames, self.unused_frames = total, total - chunks * 17
            self.source = RenderSource(
                conditions=conditions,
                image=reference,
                prompt=prompt,
                seed=seed,
                chunks=chunks,
            )
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
