"""Module that contains the schemas."""

import typing as tp
from pathlib import Path
from typing import Optional

from pydantic import BaseModel  # pylint: disable=no-name-in-module


class BaseAPIExceptionModel(BaseModel):
    """Schema for a base api exception."""

    details: str
    status_code: tp.Literal[500] = 500
    title: tp.Literal["internal server error"] = "internal server error"


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: Optional[float]


class EvaluationRequest(BaseModel):
    test_data_path: str  # The test directory containing test images


class ModelListResponse(BaseModel):
    models: list[str]


class DataDirResponse(BaseModel):
    dir_content: list[Path]


class EvaluationResponse(BaseModel):
    accuracy: float
    confusion_matrix: str
    AUC: Optional[float]
