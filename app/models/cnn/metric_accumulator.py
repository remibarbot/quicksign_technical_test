import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import Tensor


class MetricAccumulator:
    def __init__(self):
        self.losses = []
        self.preds = []
        self.targets = []

    def update(self, outputs: Tensor, targets: Tensor, loss: Tensor) -> None:
        self.losses.append(loss.item())
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        self.preds.extend(preds)
        self.targets.extend(targets.cpu().numpy())

    def compute(self) -> tuple[float, float]:
        acc = accuracy_score(self.targets, self.preds)
        avg_loss = float(np.mean(self.losses))
        return avg_loss, acc

    def reset(self) -> None:
        self.losses.clear()
        self.preds.clear()
        self.targets.clear()
