"""Centralized application configuration loaded from environment variables / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings. Values are loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider
    # SecretStr keeps the key out of repr()/print()/logs; use .get_secret_value() to read it.
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "https://api.openai.com/v1"

    # Application
    app_env: str = "development"
    log_level: str = "INFO"

    # Storage paths (relative to project root)
    database_url: str = "sqlite:///./data/test_automation.db"
    generated_tests_dir: str = "tests/generated"
    screenshots_dir: str = "screenshots"
    reports_dir: str = "reports"
    logs_dir: str = "logs"

    # Safety: generated Playwright test files may only ever be written inside this directory.
    allowed_workspace_dir: str = "tests/generated"

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Comma-separated list of allowed CORS origins. Empty by default — no
    # CORS middleware is added at all unless origins are explicitly
    # configured, and "*" is never used as a default.
    cors_origins: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        """`cors_origins` parsed into a list, e.g. "http://a.com,http://b.com" -> [...]."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_workspace_path(self) -> Path:
        """Absolute, resolved path that generated test files must stay within."""
        return (PROJECT_ROOT / self.allowed_workspace_dir).resolve()

    @property
    def reports_path(self) -> Path:
        """Absolute, resolved path to the reports directory."""
        return (PROJECT_ROOT / self.reports_dir).resolve()

    @property
    def screenshots_path(self) -> Path:
        """Absolute, resolved path to the screenshots directory."""
        return (PROJECT_ROOT / self.screenshots_dir).resolve()

    @property
    def database_path(self) -> Path:
        """Absolute, resolved path to the SQLite database file, parsed from `database_url`.

        Only the simple `sqlite:///<relative-path>` form used by this
        project is supported — there is no ORM/driver here, just sqlite3.
        """
        raw_path = self.database_url.removeprefix("sqlite:///")
        return (PROJECT_ROOT / raw_path).resolve()


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so the .env file is only parsed once."""
    return Settings()
