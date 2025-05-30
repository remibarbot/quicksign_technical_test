from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from torch import Tensor
from torchvision import datasets

from app.data.data_loading import (
    extract_hog_for_image,
    extract_hog_for_index,
    get_data_loader,
    get_hog_dataset,
    get_hog_transforms,
    get_train_transforms,
    get_val_transforms,
)
from tests.conftest import (
    create_fake_image_at_path,
    create_fake_image_folder_structure,
)


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
    create_fake_image_folder_structure(tmp_path, 1, 64)

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
    create_fake_image_folder_structure(tmp_path, 2, 64)

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


def test_extract_hog_for_image__compute_non_null_hog_features(tmp_path: Path):
    # Given
    image_size = 32
    img_path = tmp_path / "img.png"
    create_fake_image_at_path(img_path, image_size)

    # When
    features = extract_hog_for_image(img_path, image_size)

    # Then
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float64
    assert features.ndim == 1
    assert (features >= 0).all()


def test_extract_hog_for_index__compute_non_null_hog_features_on_a_image_from_a_dataset_using_hog_transforms(
    tmp_path,
):
    # Given
    img_size = 32
    root_path = tmp_path / "data"
    create_fake_image_folder_structure(root_path, n_per_class=1, size=img_size)
    dataset = datasets.ImageFolder(root_path, transform=get_hog_transforms(32))
    idx = 0

    # When
    features, label = extract_hog_for_index(idx, dataset)

    # Then
    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float64
    assert features.ndim == 1
    assert (features >= 0).all()
    assert isinstance(label, int)


def test_get_hog_dataset__creates_features_for_the_first_run_and_loads_them_for_the_second(
    tmp_path,
):
    # Given
    img_size = 32
    root_path = tmp_path / "dataset"
    create_fake_image_folder_structure(root_path, n_per_class=2, size=img_size)
    cache_name = "hog_features.npz"

    real_savez = np.savez_compressed
    # When: First run should compute and save features
    with patch("numpy.savez_compressed") as mock_savez:
        # Let the real function be called, so data is saved for next step
        mock_savez.side_effect = real_savez
        features1, labels1 = get_hog_dataset(
            root_path, img_size, cache_name=cache_name, n_jobs=2
        )

    # Then
    assert isinstance(features1, np.ndarray)
    assert isinstance(labels1, np.ndarray)
    assert features1.ndim == 2  # (n_samples, feature_dim)
    assert labels1.ndim == 1
    assert (root_path / cache_name).exists()
    mock_savez.assert_called_once()

    real_load = np.load
    # When: Second run should load from cache, not recompute
    with patch("numpy.load") as mock_load:
        mock_load.side_effect = real_load
        features2, labels2 = get_hog_dataset(
            root_path, img_size, cache_name=cache_name, n_jobs=2
        )

    # Then
    assert np.array_equal(features1, features2)
    assert np.array_equal(labels1, labels2)
    mock_load.assert_called_once()
