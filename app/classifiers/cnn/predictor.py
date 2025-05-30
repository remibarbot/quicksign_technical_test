from os import getenv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tabulate import tabulate
from tqdm import tqdm
from PIL import Image

from app.classifiers.cnn import IMAGE_SIZE
from app.classifiers.cnn.simple_cnn import SimpleCNN
from app.data.data_loading import get_data_loader, get_val_transforms


class Predictor:
    def __init__(self, checkpoint_path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(IMAGE_SIZE)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

    def predict(self, image_path: Path) -> tuple[str, float]:
        classes = {0: "Handwritten", 1: "Printed"}
        self.model.eval()
        transforms = get_val_transforms(IMAGE_SIZE)
        image = Image.open(image_path).convert("RGB")
        tensor = transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = (
                torch.sigmoid(logits).cpu().numpy().flatten()[0]
            )
            pred_class = int(prob > 0.5)
            # Confidence is probability of predicted class
            confidence = prob if pred_class == 1 else 1 - prob
        return classes[pred_class], round(float(confidence), 4)


    def evaluate(self, path_to_test_data: Path) -> dict[str, Any]:
        print(f"Evaluating on test data in : {path_to_test_data}")
        test_loader = get_data_loader(
            Path(path_to_test_data),
            batch_size=32,
            shuffle=False,
            transforms=get_val_transforms(image_size=IMAGE_SIZE),
        )
        self.model.eval()
        all_targets: list[NDArray[np.int_]] = []
        all_probs: list[NDArray[np.float64]] = []
        all_preds: list[NDArray[np.bool_]] = []

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
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

        class_names = ["Handwritten", "Printed"]
        # Confusion matrix: (true labels, predicted labels)
        cm = confusion_matrix(all_targets, all_preds_int, normalize="true")
        # ROC AUC: (true labels, predicted probabilities)
        auc = roc_auc_score(all_targets, all_probs)
        report = classification_report(
                all_targets, all_preds_int, target_names=class_names,output_dict=True
            )

        # Build table with headers
        headers = [""] + [f"Pred {name}" for name in class_names]
        table = []
        for i, row in enumerate(cm):
            table.append([f"True {class_names[i]}"] + list(row))
        cm_string = tabulate(table, headers, tablefmt="grid")

        return {
            "accuracy" : report["accuracy"],
            "confusion_matrix" : cm_string,
            "AUC": round(float(auc), 4)}
