from pathlib import Path
from unittest.mock import Mock, patch

from torch.utils.data import DataLoader

from app.models.cnn.training import IMAGE_SIZE, SimpleTrainer
from tests.conftest import create_fake_image_at_path


def test_simple_trainer__initializes_with_data_loaders_for_train_and_val(
    tmp_path: Path, monkeypatch
) -> None:
    # Given
    val_data_path = tmp_path / "val" / "class"
    train_data_path = tmp_path / "train" / "class"
    val_data_path.mkdir(parents=True, exist_ok=True)
    train_data_path.mkdir(parents=True, exist_ok=True)
    create_fake_image_at_path(
        val_data_path / "image_test.jpg", IMAGE_SIZE, mode="L"
    )
    create_fake_image_at_path(
        train_data_path / "image_test.jpg", IMAGE_SIZE, mode="L"
    )
    monkeypatch.setenv("PATH_TO_DATA", str(tmp_path))

    # When
    trainer = SimpleTrainer()

    # Then
    assert isinstance(trainer.train_loader, DataLoader)
    assert isinstance(trainer.train_loader, DataLoader)


@patch(f"{SimpleTrainer.__module__}.SimpleTrainer._validation_pass")
@patch(f"{SimpleTrainer.__module__}.SimpleTrainer._training_pass")
def test_train__calls_training_pass_and_val_pass_for_each_epochs(
    training_pass_mock: Mock,
    validation_pass_mock: Mock,
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    val_data_path = tmp_path / "val" / "class"
    train_data_path = tmp_path / "train" / "class"
    val_data_path.mkdir(parents=True, exist_ok=True)
    train_data_path.mkdir(parents=True, exist_ok=True)
    create_fake_image_at_path(
        val_data_path / "image_test.jpg", IMAGE_SIZE, mode="L"
    )
    create_fake_image_at_path(
        train_data_path / "image_test.jpg", IMAGE_SIZE, mode="L"
    )
    monkeypatch.setenv("PATH_TO_DATA", str(tmp_path))
    trainer = SimpleTrainer()
    num_epochs = 5
    training_pass_mock.side_effect = [(0.5, 0.8)] * num_epochs
    validation_pass_mock.side_effect = [
        (0.5, 0.7),
        (0.4, 0.7),
        (0.3, 0.7),
        (0.2, 0.7),
        (0.1, 0.7),
    ]

    # When
    trainer.train(num_epochs=num_epochs)

    # Then
    assert training_pass_mock.call_count == num_epochs
    assert validation_pass_mock.call_count == num_epochs


@patch(f"{SimpleTrainer.__module__}.SimpleTrainer._validation_pass")
@patch(f"{SimpleTrainer.__module__}.SimpleTrainer._training_pass")
def test_train__stops_early_when_the_val_loss_raises_two_times_in_a_row(
    training_pass_mock: Mock,
    validation_pass_mock: Mock,
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Given
    val_data_path = tmp_path / "val" / "class"
    train_data_path = tmp_path / "train" / "class"
    val_data_path.mkdir(parents=True, exist_ok=True)
    train_data_path.mkdir(parents=True, exist_ok=True)
    create_fake_image_at_path(
        val_data_path / "image_test.jpg", IMAGE_SIZE, mode="L"
    )
    create_fake_image_at_path(
        train_data_path / "image_test.jpg", IMAGE_SIZE, mode="L"
    )
    monkeypatch.setenv("PATH_TO_DATA", str(tmp_path))
    trainer = SimpleTrainer()
    num_epochs = 5
    training_pass_mock.side_effect = [(0.5, 0.8)] * num_epochs
    validation_pass_mock.side_effect = [
        (0.5, 0.7),
        (0.4, 0.7),
        (0.41, 0.7),
        (0.42, 0.7),
        (0.1, 0.7),
    ]

    expected_number_of_calls = 4

    # When
    trainer.train(num_epochs=num_epochs)

    # Then
    assert training_pass_mock.call_count == expected_number_of_calls
    assert validation_pass_mock.call_count == expected_number_of_calls
