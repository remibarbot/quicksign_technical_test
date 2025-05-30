from os import getenv
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tabulate import tabulate

from app.classifiers.svm import IMAGE_SIZE
from app.data.data_loading import extract_hog_for_image, get_hog_dataset


def train_svm_classifier(path_to_train_data: Path) -> None:
    features_training, labels_training = get_hog_dataset(
        path_to_train_data, IMAGE_SIZE
    )
    svm_classifier = make_pipeline(StandardScaler(), LinearSVC(verbose=True))
    print("Fitting SVM classifier...")
    svm_classifier.fit(features_training, labels_training)
    print("SVM trained, saving...")
    joblib.dump(
        svm_classifier,
        Path(__file__).parent / "models" / "svm_hog_classifier.joblib",
    )
    print("SVM model saved to models/svm_hog_classifier.joblib")


def evaluate_svm_classifier(
    path_to_test_data: Path, svm_model: BaseEstimator
) -> dict[str, Any]:
    class_names = ["Handwritten", "Printed"]
    features_testing, labels_testing = get_hog_dataset(
        path_to_test_data, IMAGE_SIZE
    )
    print("Evaluating SVM classifier...")
    labels_prediction = svm_model.predict(features_testing)

    report = classification_report(
        labels_testing,
        labels_prediction,
        target_names=class_names,
        output_dict=True,
    )

    cm = confusion_matrix(labels_testing, labels_prediction, normalize="true")

    # Build table with headers
    headers = [""] + [f"Pred {name}" for name in class_names]
    table = []
    for i, row in enumerate(cm):
        table.append([f"True {class_names[i]}"] + list(row))
    cm_string = tabulate(table, headers, tablefmt="grid")

    return {
        "accuracy": report["accuracy"],
        "confusion_matrix": cm_string,
        "AUC": None,
    }


def inference_svm_classifier(image_path: Path, svm_model: BaseEstimator) -> str:
    classes = {0: "Handwritten", 1: "Printed"}
    features = extract_hog_for_image(image_path, IMAGE_SIZE).reshape(1, -1)
    prediction = svm_model.predict(features)
    return classes[int(prediction[0])]


def get_svm_model() -> BaseEstimator:
    return joblib.load(
        Path(__file__).parent / "models" / "svm_hog_classifier.joblib"
    )


if __name__ == "__main__":
    path_to_data = getenv(
        "PATH_TO_TRAIN_DATA", "/home/remi/work/document_dataset"
    )
    print(f"Training on data in : {path_to_data}")
    train_svm_classifier(Path(path_to_data))
