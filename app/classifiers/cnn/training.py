from os import getenv
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from app.classifiers.cnn import IMAGE_SIZE
from app.classifiers.cnn.metric_accumulator import MetricAccumulator
from app.classifiers.cnn.simple_cnn import SimpleCNN
from app.data.data_loading import (
    get_data_loader,
    get_train_transforms,
    get_val_transforms,
)


class SimpleTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(self.device))
        self.path_to_data = getenv(
            "PATH_TO_TRAIN_DATA", "/home/remi/work/document_dataset"
        )
        print(f"Training on data in : {self.path_to_data}")
        self.model = SimpleCNN(IMAGE_SIZE).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.train_loader = get_data_loader(
            Path(self.path_to_data) / "train",
            batch_size=32,
            shuffle=True,
            transforms=get_train_transforms(image_size=IMAGE_SIZE),
        )
        self.val_loader = get_data_loader(
            Path(self.path_to_data) / "val",
            batch_size=32,
            shuffle=False,
            transforms=get_val_transforms(image_size=IMAGE_SIZE),
        )

    def _training_pass(self) -> tuple[float, float]:
        self.model.train()
        train_accum = MetricAccumulator()

        for inputs, targets in tqdm(
            self.train_loader, desc="Training", total=len(self.train_loader)
        ):
            inputs, targets = (
                inputs.to(self.device),
                targets.to(self.device).float().unsqueeze(1),
            )
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_accum.update(outputs, targets, loss)

        train_loss, train_acc = train_accum.compute()
        return train_loss, train_acc

    def _validation_pass(self) -> tuple[float, float]:
        self.model.eval()
        val_accum = MetricAccumulator()
        with torch.no_grad():
            for inputs, targets in tqdm(
                self.val_loader, desc="Validation", total=len(self.val_loader)
            ):
                inputs, targets = (
                    inputs.to(self.device),
                    targets.to(self.device).float().unsqueeze(1),
                )
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_accum.update(outputs, targets, loss)

        val_loss, val_acc = val_accum.compute()
        return val_loss, val_acc

    def train(self, num_epochs: int) -> None:
        """Train the model for num_epochs epochs.

        Will stop early if the validation loss raises two times in a row.
        """
        last_val_loss = float("inf")
        was_over_last_val_loss = False
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            train_loss, train_acc = self._training_pass()
            val_loss, val_acc = self._validation_pass()

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            if was_over_last_val_loss and val_loss > last_val_loss:
                print(
                    "Validation Loss augmented 2 times in a row, early stopping ..."
                )
                return
            was_over_last_val_loss = val_loss > last_val_loss
            last_val_loss = val_loss


if __name__ == "__main__":
    number_of_epochs = 10
    trainer = SimpleTrainer()
    trainer.train(number_of_epochs)
    torch.save(trainer.model.state_dict(), "checkpoints/simple_cnn_binary.pt")
