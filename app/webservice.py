"""Base API."""

import shutil
import tempfile
import typing as tp
from pathlib import Path

from fastapi import Body, FastAPI, File, Request, UploadFile, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.classifiers.cnn.predictor import Predictor
from app.classifiers.svm.svm_classifier import (
    evaluate_svm_classifier,
    get_svm_model,
    inference_svm_classifier,
)
from app.exceptions import (
    BaseAPIException,
)
from app.schemas import (
    DataDirResponse,
    EvaluationRequest,
    EvaluationResponse,
    ModelListResponse,
    PredictionResponse,
)


class HealthCheckResponse(BaseModel):  # pylint: disable=too-few-public-methods
    """Schema for health-check response."""

    status: str = "ok"


async def root(req: Request) -> RedirectResponse:
    """Simple redirection to '/docs' taking root_path into account.

    Args:
        req: a request made to the root path.

    Returns:
        a redirection to the docs route.
    """
    root_path = req.scope.get("root_path", "").rstrip("/")
    return RedirectResponse(root_path + "/docs")


async def health():
    """Simple health-check response."""
    return HealthCheckResponse()


def add_base_routes(
    source_app: FastAPI,
) -> None:
    """Add basic routes to a FastAPI app.

    added routes are:
      - '/health' => return {'status': 'ok'}
      - '/' => redirect to '/docs'

    Args:
        source_app: instance of a FastAPI application
    """
    # make sure we have a FastAPI app :)
    assert isinstance(source_app, FastAPI)

    # add basic health check route
    source_app.add_api_route(
        "/health",
        health,
        status_code=status.HTTP_200_OK,
        include_in_schema=True,
        response_model=HealthCheckResponse,
    )

    # add redirect route to /docs
    # not included in openAPI schema
    source_app.add_api_route(
        "/",
        root,
        include_in_schema=False,
    )


def create_app(
    debug: bool = False,
    title: str = "FastAPI",
    description: str = "FastAPI app",
    **kwargs: tp.Any,
) -> FastAPI:
    """Create a FastAPI application with basic routes.

    Args:
        debug: Run the app in debug mode. Defaults to False.
        title: Defaults to "FastAPI".
        description: Defaults to "FastAPI app".
        kwargs: other keyword arguments to pass to FastAPI app.

    Returns:
        The FastAPI app ready to be used or extended.
    """
    # FastAPI instance
    new_app = FastAPI(
        debug=debug,
        title=title,
        version="0.1.0",
        description=description,
        **kwargs,
    )
    add_base_routes(new_app)
    return new_app


app = create_app(
    title="webservice",
    description="Service to run application.",
)

# --- Loading model globally at startup ---
cnn_predictor = Predictor(
    checkpoint_path=Path(__file__).parent
    / "classifiers"
    / "cnn"
    / "checkpoints"
    / "simple_cnn_binary.pt"
)
svm_model = get_svm_model()
# --------------------------------------


EXTRA_RESPONSES = {
    **BaseAPIException.response_model(),
}


@app.get(
    "/models",
    response_model=ModelListResponse,
    responses=EXTRA_RESPONSES,
)
async def list_available_models():
    return ModelListResponse(models=["cnn", "svm"])


@app.post(
    "/evaluate/{model_type}",
    response_model=EvaluationResponse,
    responses=EXTRA_RESPONSES,
)
async def evaluate_model(
    model_type: str,
    evaluation_request: EvaluationRequest = Body(...),
):
    test_data_path = Path(evaluation_request.test_data_path)
    if not test_data_path.exists():
        raise BaseAPIException("The evaluation directory does not exist.")
    if model_type == "cnn":
        results = cnn_predictor.evaluate(test_data_path)
    elif model_type == "svm":
        results = evaluate_svm_classifier(test_data_path, svm_model)
    else:
        raise BaseAPIException(
            "Unknown model type. Run /models first to know available model types."
        )
    return results


@app.post(
    "/predict/{model_type}",
    response_model=PredictionResponse,
    responses=EXTRA_RESPONSES,
)
async def predict_image_with_model(
    model_type: str,
    file: UploadFile = File(None),
):
    confidence = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = Path(tmp.name)
    remove_after = True
    try:
        if model_type == "cnn":
            pred_class, confidence = cnn_predictor.predict(image_path)
        elif model_type == "svm":
            pred_class = inference_svm_classifier(image_path, svm_model)
        else:
            raise BaseAPIException(
                "Unknown model type. Run /models first to know available model types."
            )

        return PredictionResponse(
            predicted_class=pred_class,
            confidence=confidence,
        )
    finally:
        if file is not None and remove_after and image_path.exists():
            image_path.unlink()


@app.get(
    "/data",
    response_model=DataDirResponse,
    responses=EXTRA_RESPONSES,
)
async def list_content_of_data_directory():
    if not Path("/data").exists():
        raise BaseAPIException(
            "The data directory does not exist. "
            "Make sure to set the env variable DATA_PATH_ON_HOST "
            "before launching the service"
        )
    return DataDirResponse(dir_content=list(Path("/data").glob("*")))
