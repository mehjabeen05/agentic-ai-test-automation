"""FastAPI application entrypoint.

Run with:
    uvicorn app:app --reload

Then open:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)

This module wires the API together (CORS, routing, startup, static
frontend serving) — it contains no business logic itself. See
api/routes.py for the endpoints and api/dependencies.py for how each
existing agent/executor/repository is constructed and injected.

The dashboard (frontend/) is served as plain static files — it is not
part of the backend's business logic and has no import-level coupling to
it; the two only ever communicate over HTTP, the same as any external
client would.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from core.config import PROJECT_ROOT, get_settings
from core.database import initialize_database
from core.logger import get_logger

FRONTEND_DIR = PROJECT_ROOT / "frontend"

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    logger.info("API started")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Agentic AI Test Automation Framework",
    description=(
        "REST API for the framework built in Steps 1-10: natural-language "
        "requirements in, AI-generated and self-healing Playwright tests "
        "out, with full execution/failure/healing history in SQLite."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_settings = get_settings()
if _settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled for origins: %s", _settings.cors_origins_list)
else:
    logger.info("CORS disabled (no CORS_ORIGINS configured)")

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["health"], summary="Liveness check")
def health() -> dict:
    """Simple liveness check — does not touch the database or any agent."""
    return {"status": "ok"}


# Static dashboard assets (style.css, app.js) at /static/..., referenced
# that way from frontend/index.html. Registered after the API routes and
# /health above, so those are always matched first regardless of mount order.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")

    @app.get("/", include_in_schema=False)
    def serve_dashboard() -> FileResponse:
        """Serve the dashboard's index.html — the only backend route that returns HTML."""
        return FileResponse(FRONTEND_DIR / "index.html")
