from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


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


def get_data_loader(
    path_to_data: Path, batch_size: int, shuffle: bool, transforms
) -> DataLoader:
    train_dataset = datasets.ImageFolder(path_to_data, transform=transforms)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
