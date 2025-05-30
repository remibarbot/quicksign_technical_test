import pytest
import torch

from app.classifiers.cnn.metric_accumulator import MetricAccumulator


def test_update__adds_the_expected_values_to_the_internal_state() -> None:
    # Given
    outputs = torch.tensor([[2.0], [-2.0], [0.0]])
    targets = torch.tensor([[1.0], [0.0], [1.0]])
    loss = torch.tensor(0.7)

    accumulator = MetricAccumulator()

    expected_preds = [1, 0, 0]
    expected_targets = [1.0, 0.0, 1.0]
    expected_losses = [0.7]

    # When
    accumulator.update(outputs, targets, loss)

    # Then
    assert accumulator.preds == expected_preds
    assert accumulator.targets == expected_targets
    assert accumulator.losses == pytest.approx(expected_losses)


def test_compute__calculate_the_expected_metrics_from_internal_state() -> None:
    # Given
    outputs = torch.tensor([[10.0], [-10.0]])
    targets = torch.tensor([[1.0], [0.0]])
    loss_1 = torch.tensor(0.25)
    loss_2 = torch.tensor(0.35)
    accumulator = MetricAccumulator()
    accumulator.update(outputs, targets, loss_1)
    accumulator.update(outputs, targets, loss_2)

    expected_average_loss = 0.3
    expected_accuracy = 1.0

    # When
    average_loss, accuracy = accumulator.compute()

    # Then
    assert average_loss == pytest.approx(expected_average_loss)
    assert accuracy == pytest.approx(expected_accuracy)
