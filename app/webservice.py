"""Base API."""

import typing as tp

from fastapi import FastAPI, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.exceptions import (
    BaseAPIException,
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


EXTRA_RESPONSES = {
    **BaseAPIException.response_model(),
}


@app.post(
    "/method",
    responses=EXTRA_RESPONSES,
)
async def expectation(
    raise_error: bool,
) -> tp.Any:
    """Example of a post method.

    Args:
        raise_error: Boolean flag to determine whether to raise an exception.
    """
    if raise_error:
        raise BaseAPIException("An error occurred.")
    return {"message": "Method POST worked well."}
