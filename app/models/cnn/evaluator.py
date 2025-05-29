from os import getenv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_auc_score,
)
from tabulate import tabulate
from tqdm import tqdm

from app.data.data_loading import get_data_loader, get_val_transforms
from app.models.cnn.simple_cnn import SimpleCNN
from app.models.cnn.training import IMAGE_SIZE


class Evaluator:
    def __init__(self, checkpoint_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(IMAGE_SIZE)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.path_to_data = getenv(
            "PATH_TO_DATA", "/home/remi/work/document_dataset"
        )
        print(f"Evaluating on test data in : {self.path_to_data}")
        self.test_loader = get_data_loader(
            Path(self.path_to_data) / "test",
            batch_size=32,
            shuffle=False,
            transforms=get_val_transforms(image_size=IMAGE_SIZE),
        )


    def evaluate(self) -> None:
        self.model.eval()
        all_targets: list[NDArray[np.int_]] = []
        all_probs: list[NDArray[np.float64]] = []
        all_preds: list[NDArray[np.bool_]] = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).float().unsqueeze(1)

                logits = self.model(inputs)
                probs = torch.sigmoid(logits)

                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend((probs > 0.5).cpu().numpy())

        all_targets = [x[0] for x in all_targets]
        all_probs = [x[0] for x in all_probs]
        all_preds_int = [int(x[0]) for x in all_preds]

        # Confusion matrix: (true labels, predicted labels)
        cm = confusion_matrix(all_targets, all_preds_int, normalize="true")
        # ROC AUC: (true labels, predicted probabilities)
        auc = roc_auc_score(all_targets, all_probs)

        print("AUC:", {auc})
        # Build table with headers
        class_names = ["Handwritten", "Printed"]
        headers = [""] + [f"Pred {name}" for name in class_names]
        table = []
        for i, row in enumerate(cm):
            table.append([f"True {class_names[i]}"] + list(row))
        print(tabulate(table, headers, tablefmt="grid"))


if __name__ == "__main__":
    evaluator = Evaluator(
        checkpoint_path=Path(__file__).parent / "checkpoints/simple_cnn_binary.pt"
    )
    evaluator.evaluate()
