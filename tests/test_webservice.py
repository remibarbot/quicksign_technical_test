"""Implement tests for webservice functions. Tests files are not documented by choice."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.exceptions import BaseAPIException
from app.webservice import app
from tests.conftest import create_fake_image_at_path


@pytest.fixture(name="client")
def app_client_fixture():
    """Use the real FastAPI app for testing."""
    client = TestClient(app)
    return client


def test_base_routes_of_create_app(client: TestClient):
    """Test the health and root endpoints."""
    # Given
    expected_status_code = 200


    # When
    response_root = client.get("/")
    response_health = client.get("/health")

    # Then
    assert response_root.status_code == expected_status_code
    assert response_health.status_code == expected_status_code
    assert response_health.json() == {"status": "ok"}


def test_base_exceptions_str_representation():
    """Test the string representation of BaseAPIException."""
    # Given
    message = "Test error"

    # When
    exc = BaseAPIException(message)

    # Then
    assert (
        str(exc)
        == f"[status_code=500][title=internal server error][details={message}]"
    )


def test_base_api_exception_response():
    """Test the response() method of BaseAPIException."""
    # Given
    message = "Test error response"
    exc = BaseAPIException(message)

    expected_status_code = 500

    # When
    response = exc.response()

    # Then
    assert response.status_code == expected_status_code
    assert json.loads(response.body.decode()) == {
        "details": f"{message}",
        "status_code": expected_status_code,
        "title": "internal server error",
    }

def test_models_endpoint(client: TestClient):
    # Given
    expected_status_code = 200
    expected_models = {"models": ["cnn", "svm"]}

    # When
    response = client.get("/models")

    # Then
    assert response.status_code == expected_status_code
    assert response.json() == expected_models


def test_predict_with_invalid_model(client: TestClient, tmp_path: Path):
    """Test /predict/{model_type} with an invalid model type."""
    # Given
    img_path = tmp_path / "test_image.jpg"
    create_fake_image_at_path(img_path, 32)

    expected_error = "Unknown model type"

    with open(img_path, "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpg")}
        with pytest.raises(BaseAPIException) as exc:
            # When
            _ = client.post("/predict/invalid_model", files=files)

    # Then
    assert expected_error in str(exc)


def test_predict_cnn_with_file(client, tmp_path):
    """Test /predict/cnn with a file upload."""
    # Given
    img_path = tmp_path / "test_image.jpg"
    create_fake_image_at_path(img_path, 32)

    expected_status_code = 200

    with open(img_path, "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpg")}
        # When
        response = client.post("/predict/cnn", files=files)

    # Then
    assert response.status_code == expected_status_code
    assert "predicted_class" in response.json()
    assert "confidence" in response.json()


def test_predict_svm_with_file(client, tmp_path):
    """Test /predict/svm with a file upload."""
    # Given
    img_path = tmp_path / "test_image.jpg"
    create_fake_image_at_path(img_path, 32)

    expected_status_code = 200

    with open(img_path, "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpg")}
        # When
        response = client.post("/predict/svm", files=files)

    # Then
    assert response.status_code == expected_status_code
    assert "predicted_class" in response.json()
    assert "confidence" in response.json()


def test_evaluate_with_nonexistent_path(client: TestClient):
    """Test /evaluate/cnn with a nonexistent test data path."""
    # Given
    payload = {"test_data_path": "/nonexistent/path/for/testing"}

    expected_error = "does not exist"

    # When
    with pytest.raises(BaseAPIException) as exc:
        _ = client.post("/evaluate/cnn", json=payload)

    # Then
    assert expected_error in str(exc)


def test_evaluate_cnn(client: TestClient, tmp_path: Path):
    """Test /evaluate/cnn endpoint."""
    # Given
    test_data_path = tmp_path / "fake_eval_dir" / "test"
    handwritten_path = test_data_path / "handwritten"
    printed_path = test_data_path / "printed"
    handwritten_path.mkdir(parents=True, exist_ok=True)
    printed_path.mkdir(parents=True, exist_ok=True)
    create_fake_image_at_path(handwritten_path / "image_1.jpg", 128)
    create_fake_image_at_path(printed_path / "image_1.jpg", 128)
    payload = {"test_data_path": str(test_data_path)}

    expected_response_code = 200

    # When
    response = client.post("/evaluate/cnn", json=payload)

    # Then
    assert response.status_code == expected_response_code

    assert "accuracy" in response.json()
    assert "confusion_matrix" in response.json()
    assert "AUC" in response.json()


def test_evaluate_svm(client: TestClient, tmp_path: Path):
    """Test /evaluate/svm endpoint."""
    # Given
    test_data_path = tmp_path / "fake_eval_dir" / "test"
    handwritten_path = test_data_path / "handwritten"
    printed_path = test_data_path / "printed"
    handwritten_path.mkdir(parents=True, exist_ok=True)
    printed_path.mkdir(parents=True, exist_ok=True)
    create_fake_image_at_path(handwritten_path / "image_1.jpg", 128)
    create_fake_image_at_path(printed_path / "image_1.jpg", 128)
    payload = {"test_data_path": str(test_data_path)}

    expected_response_code = 200

    # When
    response = client.post("/evaluate/svm", json=payload)

    # Then
    assert response.status_code == expected_response_code

    assert "accuracy" in response.json()
    assert "confusion_matrix" in response.json()
    assert "AUC" in response.json()


def test_data_endpoint_with_existing_dir(client: TestClient, tmp_path: Path, monkeypatch):
    """Test /data endpoint when /data exists."""
    # Given
    # Patch Path("/data").exists to True, and glob to return a list
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    monkeypatch.setattr("pathlib.Path.glob", lambda self, pat: [tmp_path / "file1.txt", tmp_path / "file2.txt"])

    # When
    response = client.get("/data")

    # Then
    assert response.status_code == 200
    assert "dir_content" in response.json()
    assert isinstance(response.json()["dir_content"], list)


def test_data_endpoint_with_missing_dir(client, monkeypatch):
    """Test /data endpoint when /data does not exist."""
    # Given
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    expected_error = "does not exist"

    # When
    with pytest.raises(BaseAPIException) as exc:
        _ = client.get("/data")

    # Then
    assert expected_error in str(exc)