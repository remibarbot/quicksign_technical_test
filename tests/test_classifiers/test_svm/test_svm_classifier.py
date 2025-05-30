from pathlib import Path

from sklearn.pipeline import Pipeline

from app.classifiers.svm.svm_classifier import (
    evaluate_svm_classifier,
    get_svm_model,
    inference_svm_classifier,
)
from tests.conftest import (
    create_fake_image_at_path,
    create_fake_image_folder_structure,
)


def test_get_svm_model__loads_the_model(tmp_path: Path):
    # When
    loaded = get_svm_model()

    # Then
    assert isinstance(loaded, Pipeline)


def test_evaluate_svm_classifier__returns_evaluation_metrics_given_a_real_test_folder(
    tmp_path: Path,
):
    # Given
    svm = get_svm_model()
    create_fake_image_folder_structure(tmp_path / "test", 2, 128)

    # When
    metrics = evaluate_svm_classifier(tmp_path / "test", svm)

    # Then
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["confusion_matrix"], str)
    assert metrics["AUC"] is None


def test_inference_svm_classifier__predicts_one_of_the_expected_class_given_an_image(
    tmp_path: Path,
):
    # Given
    svm = get_svm_model()
    test_image = tmp_path / "test_img.jpg"
    create_fake_image_at_path(test_image, size=128)

    # When
    result = inference_svm_classifier(test_image, svm)

    # Then
    assert result in {"Handwritten", "Printed"}
