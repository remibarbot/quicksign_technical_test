from pathlib import Path

from app.classifiers.cnn.predictor import Predictor
from tests.conftest import (
    create_fake_image_at_path,
    create_fake_image_folder_structure,
)


def test_predict__predicts_a_class_and_a_confidence_from_an_image(tmp_path: Path):
    # Given
    checkpoint_path = (
        Path(__file__).parents[3]
        / "app"
        / "classifiers"
        / "cnn"
        / "checkpoints"
        / "simple_cnn_binary.pt"
    )
    predictor = Predictor(checkpoint_path)

    test_img = tmp_path / "test_img.jpg"
    create_fake_image_at_path(test_img, size=128)

    # When
    pred_class, confidence = predictor.predict(test_img)

    # Then
    assert isinstance(pred_class, str)
    assert pred_class in {"Handwritten", "Printed"}
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1


def test_evaluate__returns_evaluation_metrics_given_a_real_test_folder(
    tmp_path: Path,
):
    # Given
    checkpoint_path = (
        Path(__file__).parents[3]
        / "app"
        / "classifiers"
        / "cnn"
        / "checkpoints"
        / "simple_cnn_binary.pt"
    )
    predictor = Predictor(checkpoint_path)

    # Create a test dataset directory with both classes and images
    create_fake_image_folder_structure(tmp_path / "test", 2, 128)

    # When
    metrics = predictor.evaluate(tmp_path / "test")

    # Then
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "AUC" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["confusion_matrix"], str)
    assert isinstance(metrics["AUC"], float)
