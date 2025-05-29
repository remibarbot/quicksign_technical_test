from pathlib import Path

import pytest
from torch import Tensor

from app.data.data_loading import (
    get_data_loader,
    get_train_transforms,
    get_val_transforms,
)
from tests.conftest import create_fake_image_at_path


def test_get_data_loader__raises_file_not_found_when_the_given_data_path_is_empty(
    tmp_path: Path,
) -> None:
    # Given
    expected_exception_info = "Couldn't find any class folder in"
    batch_size = 2
    transforms = get_train_transforms(32)
    shuffle = True

    # Then
    with pytest.raises(FileNotFoundError) as excinfo:
        # When
        _ = get_data_loader(tmp_path, batch_size, transforms, shuffle)

    # Then
    assert expected_exception_info in str(excinfo.value)


def test_get_data_loader__returns_an_iterable_data_loader_when_using_train_transforms(
    tmp_path: Path,
) -> None:
    # Given
    handwritten_dir = tmp_path / "handwritten"
    handwritten_dir.mkdir()
    printed_dir = tmp_path / "printed"
    printed_dir.mkdir()
    create_fake_image_at_path(handwritten_dir / "image.jpg", 64, mode="RGB")
    create_fake_image_at_path(printed_dir / "image.jpg", 64, mode="RGB")

    batch_size = 2
    input_image_size = 32
    transforms = get_train_transforms(input_image_size)
    shuffle = False

    # When
    dataloader = get_data_loader(tmp_path, batch_size, shuffle, transforms)
    images, labels = next(iter(dataloader))

    # Then
    assert isinstance(images, Tensor)
    assert images.shape == (2, 1, 32, 32)
    assert labels.shape == (2,)
    assert -1.0 < images.mean().item() < 1.0


def test_get_data_loader__returns_an_iterable_data_loader_when_using_validation_transforms(
    tmp_path: Path,
) -> None:
    # Given
    handwritten_dir = tmp_path / "handwritten"
    handwritten_dir.mkdir()
    printed_dir = tmp_path / "printed"
    printed_dir.mkdir()
    create_fake_image_at_path(handwritten_dir / "image.jpg", 64, mode="RGB")
    create_fake_image_at_path(printed_dir / "image.jpg", 64, mode="RGB")
    create_fake_image_at_path(handwritten_dir / "image2.jpg", 64, mode="L")
    create_fake_image_at_path(printed_dir / "image2.jpg", 64, mode="L")

    batch_size = 4
    input_image_size = 28
    transforms = get_val_transforms(input_image_size)
    shuffle = True

    # When
    dataloader = get_data_loader(tmp_path, batch_size, shuffle, transforms)
    images, labels = next(iter(dataloader))

    # Then
    assert isinstance(images, Tensor)
    assert images.shape == (4, 1, 28, 28)
    assert labels.shape == (4,)
    assert -1.0 < images.mean().item() < 1.0
