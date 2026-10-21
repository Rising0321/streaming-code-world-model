"""Read the semantic reference frames matching a decoded native chunk."""

from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = np.array(
    [
        [0, 0, 0],
        [135, 206, 235],
        [30, 90, 200],
        [150, 120, 80],
        [90, 90, 90],
        [60, 160, 60],
        [200, 120, 80],
        [220, 200, 60],
        [230, 40, 40],
        [200, 80, 200],
        [40, 220, 220],
        [250, 250, 250],
    ],
    dtype=np.uint8,
)


def reference_frames(conditions: Path, chunk: int) -> np.ndarray:
    """Colour semantic frames 17*k through 17*k+16 at their source resolution."""
    if chunk < 0:
        raise ValueError("Expected a nonnegative emitted chunk index")
    reference = np.empty((17, 192, 336, 3), dtype=np.uint8)
    for offset in range(17):
        with Image.open(
            conditions / f"{17 * chunk + offset:06d}.semantic_id.png"
        ) as image:
            ids = np.asarray(image)
            if ids.shape != (192, 336) or image.mode != "L" or ids.max() > 11:
                raise ValueError("Invalid semantic reference frame")
            reference[offset] = PALETTE[ids]
    return reference
