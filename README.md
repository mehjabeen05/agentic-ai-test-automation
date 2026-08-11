![Uploading WhatsApp Image 2026-08-12 at 12.28.20 AM.jpeg…]()

# Agentic AI Test Automation Framework

## Overview

An AI-powered test automation framework: describe a software requirement in
plain English, and the system generates structured test cases, writes
executable Playwright tests, runs them, analyzes any failures with an LLM,
and can automatically recover from broken UI selectors — all backed by a
REST API, a SQLite history store, and a browser dashboard.

Every LLM output is treated as **untrusted** until it passes strict
validation: structured data goes through Pydantic, generated code goes
through AST-based static analysis, and nothing an LLM produces is ever
executed, evaluated, or trusted directly. See [Security](#security) below.

## Key Features

- **Natural-language requirements → structured test cases** — a plain
  English requirement is turned into a validated `RequirementAnalysis` and
  then a deduplicated set of positive/negative/boundary/validation/security
  `TestCase` objects.
- **AI-generated Playwright code, safety-checked before it ever touches
  disk** — every generated test is parsed with Python's `ast` module and
  rejected if it imports anything dangerous, calls `eval`/`exec`/
  `os.system`, deletes files, or hardcodes credentials.
- **Sandboxed execution** — generated tests only ever run from
  `tests/generated/`, via a fixed argument list passed to `subprocess.run`
  with `shell=False` — never a shell string.
- **LLM-assisted failure analysis** — a failed run is explained (failure
  type, root cause, suggested fix, confidence) from redacted, truncated
  execution context, not the raw prompt.
- **Controlled self-healing** — when a test fails because a selector no
  longer matches the page, the framework proposes replacement selectors and
  proves each one against the live browser DOM before ever using it. The
  LLM never touches test code, only candidate selector strings.
- **Full history in SQLite** — every requirement, test case, execution,
  failure analysis, and healing attempt is persisted and queryable.
- **REST API + browser dashboard** — the entire workflow above is reachable
  over HTTP and usable from a plain HTML/CSS/JS dashboard, with no
  framework or build step.

## Architecture

```
User → Dashboard → FastAPI → AI Agents → Playwright → Test Runner
                                                            │
                                                            ▼
                                                   Failure Analysis
                                                            │
                                                            ▼
                                                     Self-Healing
                                                            │
                                                            ▼
                                                        SQLite
```

```
frontend/  ── static HTML/CSS/JS dashboard (served by FastAPI)
   │  fetch()
   ▼
app.py + api/  ── FastAPI routes, dependency injection, no business logic
   │
   ▼
agents/  ── RequirementAgent, TestCaseGeneratorAgent,
             FailureAnalysisAgent, SelectorHealingAgent
generators/  ── PlaywrightCodeGenerator + AST code_validator
executor/  ── TestRunner (sandboxed subprocess), HealingExecutor
   │
   ▼
core/  ── config, models (Pydantic), llm_client, database, repositories
   │
   ▼
SQLite (data/test_automation.db)
```

## Tech Stack

- **Language:** Python 3.11+ (developed and tested on 3.12)
- **API:** FastAPI + Uvicorn
- **Validation:** Pydantic v2, pydantic-settings
- **Browser automation:** Playwright (sync API), pytest-playwright
- **Testing:** PyTest
- **LLM:** any OpenAI-compatible chat completions API (OpenAI, Azure OpenAI,
  a local proxy, etc.), via the official `openai` SDK
- **Database:** SQLite (Python's built-in `sqlite3`, no ORM)
- **Frontend:** plain HTML, CSS, vanilla JavaScript — no framework
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions
- **Linting:** ruff

## Installation

Requires Python 3.11+ and Git. Clone or open this repository, then follow
[Local Setup](#local-setup) below.

## Local Setup

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

For linting during development, also install the dev extras:

```bash
pip install -r requirements-dev.txt
```

## Playwright installation

```bash
python -m playwright install chromium
```

Only the Chromium browser is required — the project never uses Firefox or
WebKit.

## Environment configuration

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and fill in your own values. `.env` is gitignored and is never
committed — nothing in this project ever asks you to paste a real API key
into chat or into a tracked file.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_API_KEY` | Your LLM provider's API key | *(empty — required for real LLM calls)* |
| `LLM_MODEL` | Model name to request | `gpt-4o-mini` |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `API_HOST` | Host the API binds to locally | `127.0.0.1` |
| `API_PORT` | Port the API binds to locally | `8000` |
| `CORS_ORIGINS` | Comma-separated allowed origins | *(empty — CORS disabled)* |

See `.env.example` for the complete list, including storage paths and the
sandboxed-workspace setting. Every value is read from the environment via
`core/config.py`; the same variables work identically whether they come
from a local `.env` file or from real environment variables (e.g. in
Docker or CI) — see [Docker](#docker).

## Run application

```bash
uvicorn app:app --reload
```

## Dashboard

Open **http://127.0.0.1:8000/** — the dashboard is served by FastAPI itself,
at the same origin as the API, so no extra configuration is needed locally.

## API Documentation

- Swagger UI: **http://127.0.0.1:8000/docs**
- ReDoc: **http://127.0.0.1:8000/redoc**

## Running tests

```bash
pytest -v
```

Unit tests never require a real `LLM_API_KEY` — every LLM-driven test
injects a fake `LLMClient` test double instead of calling a real provider.
A small number of browser-driven tests (`tests/test_login.py`,
`tests/test_healing_agent.py`'s integration test, `tests/test_frontend.py`)
need Chromium installed (see [Playwright installation](#playwright-installation)).

## Docker

Build and run with Docker Compose (recommended — also wires up persistent
volumes and the health check):

```bash
docker compose up --build
```

Or with the Docker CLI directly:

```bash
docker build -t agentic-test-automation .
docker run -p 8000:8000 --env-file .env agentic-test-automation
```

The image installs only the Chromium browser, runs as a non-root user, and
never bakes in `.env` or any secret — configuration is supplied entirely
through environment variables at container-start time (see
`docker-compose.yml` and `.dockerignore`).

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request: it checks
out the repository, sets up Python 3.12, installs dependencies (including
Chromium), lints with `ruff check .`, and runs the full test suite with
`pytest -v`. No `LLM_API_KEY` is configured in CI — every test that would
otherwise need one uses a mocked LLM client, so the pipeline never depends
on a real provider or a real key.

## Example Workflow

```
"User should be able to login successfully."
        │
        ▼
Requirement Agent  ──▶  RequirementAnalysis (structured)
        │
        ▼
Test Case Generator Agent  ──▶  Test Cases (positive/negative/boundary/...)
        │
        ▼
Playwright Code Generator  ──▶  Playwright Code (AST-validated)
        │
        ▼
Code Validator  ──▶  Validation (safe to save/run, or rejected)
        │
        ▼
Test Runner  ──▶  Execution (sandboxed subprocess)
        │
        ▼
Failure Analysis Agent  ──▶  Failure Analysis (if it failed)
        │
        ▼
Self-Healing  ──▶  Selector recovery (if the failure was selector-related)
        │
        ▼
Report  ──▶  Persisted to SQLite, viewable in the dashboard
```

## Self-Healing Example

```
Original selector:  #login-button
Failure:             selector_not_found (0 matches on the live page)
Candidates proposed: ["#login-button", "button.login", "[data-testid='login']"]
Browser validation:  "#login-button" → 0 matches (rejected)
                      "button.login" → 1 match (valid)
                      "[data-testid='login']" → 0 matches (rejected)
Recovered selector:  button.login
```

A candidate selector is never trusted because the LLM proposed it. Every
candidate is checked against the real, live browser DOM
(`page.locator(selector).count()`) before it can be selected, and only a
selector matching **exactly one** element is ever used for the retry. The
LLM can only ever propose plain selector strings — never code, and never
the action that gets performed on retry.

## Security

- **API keys are never hardcoded.** `LLM_API_KEY` (and every other secret)
  is read from an environment variable via `core/config.py`, stored as a
  Pydantic `SecretStr` so it cannot leak through `print()`, a log line, or
  a `repr()`. `.env` is gitignored; `.env.example` contains placeholders only.
  Docker never copies `.env` into the image (see `.dockerignore`).
- **LLM output is always untrusted** until it passes explicit validation:
  structured responses are parsed and validated with Pydantic; generated
  Playwright code is parsed with Python's `ast` module and statically
  checked — it is never executed, evaluated, or imported at that stage.
- **Generated code is rejected, not sanitized, when unsafe** — forbidden
  imports (`subprocess`, `socket`, `pickle`, `sys`, non-Playwright network
  libraries), dangerous calls (`eval`, `exec`, `os.system`), file-deletion
  calls, and likely hardcoded credentials all fail validation outright.
- **Arbitrary shell commands are never possible.** Generated tests execute
  through a fixed argument list via `subprocess.run(..., shell=False)` —
  there is no shell string anywhere in that path.
- **Generated test execution is restricted to one sandboxed directory**
  (`tests/generated/`), enforced by resolving the path and checking its
  parent directory before the file is even opened.
- **Credentials are never hardcoded** — generated tests read credentials
  and target URLs from environment variables with safe placeholder
  defaults, never literal values.
- **Self-healing only ever changes a selector at runtime**, in memory, for
  that one retry — it never rewrites a test file, and every candidate must
  be independently proven against the live DOM first.

## Limitations

- LLM output can be incorrect, incomplete, or overly confident — every
  agent's output (test cases, failure analyses, healing suggestions) should
  be treated as a well-informed starting point, not a verified result.
- Self-healing is limited to selector recovery — it cannot fix incorrect
  test logic, bad assertions, or environment problems.
- Screenshot/image analysis is not implemented — failure analysis is
  text-only; a captured screenshot's path is recorded but never inspected
  by the LLM.
- No production-grade authentication or multi-user support — this is a
  single-user, single-machine framework as built.
- The self-healing API endpoint (`/api/v1/tests/heal`) launches a fresh
  browser page per request rather than reusing the original failed
  session, so without an explicit `url` it typically has no live DOM to
  validate candidates against.

## Future Improvements

- Visual AI testing (screenshot/image-based failure analysis)
- Multi-browser parallel execution (Firefox, WebKit, sharded runs)
- Deeper test analytics and trend reporting
- User authentication and multi-user support
- Cloud-based test execution
- Kubernetes deployment
- Slack/Teams notifications for failures and healing events
- Advanced visual regression testing
