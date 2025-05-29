from pathlib import Path

import numpy as np
from PIL import Image


def create_fake_image_at_path(path: Path, size: int, mode: str = "L") -> None:
    arr = np.random.randint(
        0, 256, (size, size) + ((1,) if mode == "L" else (3,)), dtype=np.uint8
    )
    img = Image.fromarray(arr.squeeze(), mode)
    img.save(path)
