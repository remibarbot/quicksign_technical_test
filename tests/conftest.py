from pathlib import Path

import numpy as np
from PIL import Image


def create_fake_image_at_path(path: Path, size: int, mode: str = "L") -> None:
    arr = np.random.randint(
        0, 256, (size, size) + ((1,) if mode == "L" else (3,)), dtype=np.uint8
    )
    img = Image.fromarray(arr.squeeze(), mode)
    img.save(path)


def create_fake_image_folder_structure(root_path: Path, n_per_class=2, size=64):
    (root_path / "handwritten").mkdir(parents=True, exist_ok=True)
    (root_path / "printed").mkdir(parents=True, exist_ok=True)
    for i in range(n_per_class):
        create_fake_image_at_path(
            root_path / "handwritten" / f"hw_{i}.jpg", size, "L"
        )
        create_fake_image_at_path(
            root_path / "printed" / f"pr_{i}.jpg", size, "L"
        )
