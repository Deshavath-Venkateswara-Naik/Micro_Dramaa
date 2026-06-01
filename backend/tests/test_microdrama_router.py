"""Integration tests for the Microdrama Generator FastAPI router.

The backend imports its packages as `services.X` / `routers.X` (the app runs
from the `backend/` directory), so we ensure that directory is importable here.
"""
import os
import sys

# Make the backend package root (parent of this tests/ dir) importable so that
# `from services.microdrama_generator import MicrodramaGenerator` resolves the
# same way it does when the FastAPI app runs.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from services.microdrama_generator import MicrodramaGenerator  # noqa: E402

from unittest.mock import MagicMock, patch  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from routers import microdrama  # noqa: E402


def _make_client():
    """Build an isolated app around the microdrama router only.

    We mount the router directly (rather than importing main.py) so the test
    avoids main.py's heavy transitive imports and its StaticFiles mount, which
    requires the real storage/ directory.
    """
    app = FastAPI()
    app.include_router(microdrama.router, prefix="/api/v1")
    return TestClient(app)


# --- Test 1: valid video_id returns the service envelope (Req 8.3, 9.1) -----

def test_generate_microdrama_valid_returns_envelope():
    """A valid request returns the service's Candidates_Output unchanged."""
    envelope = {
        "video_id": "MOV_TEST",
        "status": "completed",
        "microdrama_candidates": [
            {
                "title": "A",
                "start_time": "00:00:05.000",
                "end_time": "00:00:45.000",
                "duration_seconds": 40.0,
                "opening_hook": "hook",
                "central_conflict": "conflict",
                "cliffhanger_ending": "cliff",
                "retention_score": 80,
                "characters_present": ["X"],
                "included_scene_numbers": [1],
            }
        ],
        "excluded_scenes": [],
        "excluded_candidates": [],
    }

    client = _make_client()
    with patch("routers.microdrama.os.path.exists", return_value=True), patch(
        "routers.microdrama.MicrodramaGenerator"
    ) as mock_generator_cls:
        instance = mock_generator_cls.return_value
        instance.generate.return_value = envelope

        response = client.post("/api/v1/generate-microdrama", json={"video_id": "MOV_TEST"})

    assert response.status_code == 200
    assert response.json() == envelope
    # The router constructed the service and asked it to generate for our id.
    assert mock_generator_cls.call_count == 1
    instance.generate.assert_called_once_with(video_id="MOV_TEST")


# --- Test 2: missing video_id -> 422 (Req 9.2) ------------------------------

def test_generate_microdrama_missing_video_id_returns_422():
    """An omitted video_id is a client validation error (pydantic)."""
    client = _make_client()
    with patch("routers.microdrama.os.path.exists", return_value=True), patch(
        "routers.microdrama.MicrodramaGenerator"
    ) as mock_generator_cls:
        response = client.post("/api/v1/generate-microdrama", json={})

    assert response.status_code == 422
    # Validation happens before any service work.
    mock_generator_cls.assert_not_called()


# --- Test 3: non-existent video_id -> 404 identifying it (Req 9.3) ----------

def test_generate_microdrama_unknown_video_id_returns_404():
    """A missing storage directory yields a 404 naming the video_id."""
    client = _make_client()
    with patch("routers.microdrama.os.path.exists", return_value=False), patch(
        "routers.microdrama.MicrodramaGenerator"
    ) as mock_generator_cls:
        response = client.post(
            "/api/v1/generate-microdrama", json={"video_id": "DOES_NOT_EXIST"}
        )

    assert response.status_code == 404
    assert "DOES_NOT_EXIST" in response.json()["detail"]
    # No generation attempted when the directory is absent.
    mock_generator_cls.assert_not_called()


# --- Test 4: service reports failure -> 500 with detail (Error Handling) -----

def test_generate_microdrama_service_failed_returns_500():
    """A failed service result maps to HTTP 500 carrying the error detail."""
    client = _make_client()
    with patch("routers.microdrama.os.path.exists", return_value=True), patch(
        "routers.microdrama.MicrodramaGenerator"
    ) as mock_generator_cls:
        instance = mock_generator_cls.return_value
        instance.generate.return_value = {
            "video_id": "MOV_TEST",
            "status": "failed",
            "error": "some failure",
        }

        response = client.post("/api/v1/generate-microdrama", json={"video_id": "MOV_TEST"})

    assert response.status_code == 500
    assert "some failure" in response.json()["detail"]


# === Smoke Tests (Task 10.4) ================================================
# Lightweight checks that the service wires up the Vertex AI genai client the
# way the design prescribes, and that the router exposes its path under the
# /api/v1 prefix exactly as main.py registers it. These are example-based; no
# property-based testing here.


# --- Smoke 1: client construction uses vertexai + GCP env vars (Req 4.1) ----

def test_microdrama_generator_constructs_vertex_client_from_env(tmp_path):
    """`MicrodramaGenerator.__init__` builds the genai client with
    `vertexai=True` and project/location pulled from the GCP_PROJECT_ID /
    GCP_LOCATION environment variables (Req 4.1)."""
    with patch("services.microdrama_generator.genai.Client") as mock_client, patch.dict(
        os.environ, {"GCP_PROJECT_ID": "proj-test", "GCP_LOCATION": "us-test"}
    ):
        gen = MicrodramaGenerator(output_base_dir=str(tmp_path))

    # The Vertex AI client was constructed exactly once with the expected kwargs.
    mock_client.assert_called_once_with(
        vertexai=True, project="proj-test", location="us-test"
    )
    # Successful construction wires the client onto the instance (not None).
    assert gen.llm_client is mock_client.return_value


# --- Smoke 2: route registered under the /api/v1 prefix (Req 9.4) -----------

def test_microdrama_route_registered_under_api_v1_prefix():
    """Including the microdrama router under the `/api/v1` prefix exposes
    `/api/v1/generate-microdrama`, mirroring main.py's registration (Req 9.4)."""
    app = FastAPI()
    app.include_router(microdrama.router, prefix="/api/v1")

    registered_paths = [route.path for route in app.routes]
    assert "/api/v1/generate-microdrama" in registered_paths
