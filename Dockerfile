# Lightweight Python base image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Install Python dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install only the Chromium browser (no Firefox/WebKit) plus its OS-level
# dependencies. PLAYWRIGHT_BROWSERS_PATH is set to a fixed, user-independent
# location so the browser is still found later when running as a non-root user.
RUN python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

# Non-root runtime user (Docker security best practice).
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser . .

# Runtime data directories — created here so they exist with the right
# ownership even before the app writes to them on first run.
RUN mkdir -p data logs reports/execution reports/healing screenshots tests/generated \
    && chown -R appuser:appuser data logs reports screenshots tests/generated

USER appuser

EXPOSE 8000

# Matches the app's own liveness check (GET /health -> {"status": "ok"}).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Configuration (LLM_API_KEY, LLM_MODEL, API_HOST, API_PORT, CORS_ORIGINS, ...)
# is supplied via environment variables at `docker run`/`docker compose` time —
# never baked into the image. See .env.example for the full list.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
