from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm


def get_train_transforms(image_size: int):
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize((image_size, image_size)),
            # v2.RandomResizedCrop(size=(image_size, image_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(degrees=20),
            v2.ToTensor(),
            v2.RandomErasing(),
            v2.Normalize([0.5], [0.5]),
        ]
    )


def get_val_transforms(image_size: int):
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize((image_size, image_size)),
            v2.ToTensor(),
            v2.Normalize([0.5], [0.5]),
        ]
    )


def get_hog_transforms(image_size: int):
    return v2.Compose(
        [
            v2.Grayscale(),
            v2.Resize((image_size, image_size)),
        ]
    )


def get_data_loader(
    path_to_data: Path, batch_size: int, shuffle: bool, transforms
) -> DataLoader:
    train_dataset = datasets.ImageFolder(path_to_data, transform=transforms)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


def extract_hog_for_image(
    image_path: Path, image_size: int
) -> NDArray[np.float64]:
    image = imread(image_path)
    if image.ndim == 3:
        image = rgb2gray(image)
    image = resize(image, (image_size, image_size))
    features = hog(
        image,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
    )
    features_array = np.asarray(features, dtype=np.float64)
    return features_array


def extract_hog_for_index(
    index: int, dataset: Dataset
) -> tuple[NDArray[np.float64], int]:
    img, label = dataset[index]
    np_img = np.array(img) / 255.0
    features = hog(
        np_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm="L2-Hys",
    )
    features_array = np.asarray(features, dtype=np.float64)
    return features_array, label


def get_hog_dataset(
    path_to_data: Path,
    image_size: int,
    cache_name: str = "hog_features.npz",
    n_jobs: int = 8,
):
    # Check for saved features first
    if (path_to_data / cache_name).exists():
        print(f"Loading precomputed HOG dataset from {cache_name}")
        with np.load((path_to_data / cache_name)) as data:
            return data["features"], data["labels"]

    # Otherwise, compute features
    dataset = datasets.ImageFolder(
        path_to_data, transform=get_hog_transforms(image_size)
    )

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(extract_hog_for_index)(idx, dataset)
        for idx in tqdm(
            range(len(dataset)),
            desc="Submitting HOG extraction jobs...",
            total=len(dataset),
        )
    )
    all_features, all_labels = zip(*results)
    all_features_np = np.array(all_features)
    all_labels_np = np.array(all_labels)

    print("Saving HOG dataset ...")
    np.savez_compressed(
        path_to_data / cache_name, features=all_features_np, labels=all_labels_np
    )
    print(f"Saved HOG dataset to {cache_name}")
    return all_features_np, all_labels_np
