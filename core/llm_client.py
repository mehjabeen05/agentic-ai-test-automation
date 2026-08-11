"""Provider-independent LLM client abstraction.

Agents should depend only on the `LLMClient` interface and `get_llm_client()`
factory in this module — never on the OpenAI SDK directly. Swapping providers
later (Azure OpenAI, Anthropic, a local model, ...) means adding a new
`LLMClient` subclass and updating the factory, with no changes required in
any agent code.

LLM output is untrusted text: this module returns a plain string and never
parses or executes it.
"""

from abc import ABC, abstractmethod

import httpx
import openai
from pydantic import BaseModel, SecretStr

from core.config import Settings, get_settings
from core.logger import get_logger

logger = get_logger(__name__)


class LLMClientError(Exception):
    """Base exception for all LLM client failures."""


class LLMConfigurationError(LLMClientError):
    """Raised when the client is misconfigured, e.g. a missing API key."""


class LLMAuthenticationError(LLMClientError):
    """Raised when the provider rejects the configured API key."""


class LLMRequestError(LLMClientError):
    """Raised for network failures, timeouts, rate limits, or provider-side errors."""


class LLMResponseError(LLMClientError):
    """Raised when the provider returns an empty or malformed response."""


class LLMClientConfig(BaseModel):
    """Validated configuration for an LLM client.

    `api_key` is a SecretStr so it never appears in plain text if this model
    is printed, logged, or included in a traceback.
    """

    api_key: SecretStr
    model: str
    base_url: str
    timeout_seconds: float = 30.0
    max_retries: int = 2

    @classmethod
    def from_settings(cls, settings: Settings) -> "LLMClientConfig":
        """Build client configuration from the application's Settings object."""
        return cls(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
        )

    def has_api_key(self) -> bool:
        """Whether a non-empty API key was configured."""
        return bool(self.api_key.get_secret_value().strip())


class LLMClient(ABC):
    """Abstract interface every LLM provider implementation must satisfy."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send `prompt` to the LLM and return its raw text response.

        Raises:
            LLMConfigurationError: the client is missing required configuration.
            LLMAuthenticationError: the provider rejected the API key.
            LLMRequestError: a network or provider-side error occurred.
            LLMResponseError: the provider returned an empty or malformed response.
        """
        raise NotImplementedError


class OpenAILLMClient(LLMClient):
    """LLMClient implementation backed by any OpenAI-compatible chat completions API."""

    def __init__(self, config: LLMClientConfig) -> None:
        self._config = config
        # The SDK client can be constructed even without a real key; generate()
        # checks for a usable key before ever making a network call.
        self._client = openai.OpenAI(
            api_key=config.api_key.get_secret_value() or "not-configured",
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def generate(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        if not self._config.has_api_key():
            logger.error("LLM generate() called with no LLM_API_KEY configured")
            raise LLMConfigurationError(
                "LLM_API_KEY is not set. Add it to your .env file (see .env.example)."
            )

        logger.info(
            "Sending prompt to LLM (model=%s, prompt_length=%d)",
            self._config.model,
            len(prompt),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except openai.AuthenticationError as exc:
            logger.error("LLM authentication failed: the API key was rejected")
            raise LLMAuthenticationError(
                "The configured LLM_API_KEY was rejected by the provider."
            ) from exc
        except openai.RateLimitError as exc:
            logger.error("LLM request failed: rate limit exceeded")
            raise LLMRequestError("LLM provider rate limit exceeded. Try again later.") from exc
        except openai.APITimeoutError as exc:
            logger.error("LLM request timed out")
            raise LLMRequestError("Timed out waiting for the LLM provider.") from exc
        except openai.APIConnectionError as exc:
            logger.error("LLM request failed: could not connect to provider")
            raise LLMRequestError(f"Could not reach the LLM provider: {exc}") from exc
        except openai.APIStatusError as exc:
            logger.error("LLM provider returned an error status: %s", exc.status_code)
            raise LLMRequestError(
                f"LLM provider returned HTTP {exc.status_code}."
            ) from exc
        except openai.APIError as exc:
            logger.error("LLM provider error: %s", type(exc).__name__)
            raise LLMRequestError(f"LLM provider error: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.error("LLM transport error: %s", type(exc).__name__)
            raise LLMRequestError(f"Transport error calling LLM provider: {exc}") from exc

        try:
            message_content = response.choices[0].message.content
        except (AttributeError, IndexError) as exc:
            logger.error("LLM returned a malformed response structure")
            raise LLMResponseError("The LLM response was malformed.") from exc

        if not message_content or not message_content.strip():
            logger.error("LLM returned an empty response")
            raise LLMResponseError("The LLM response was empty.")

        logger.info("Received LLM response (length=%d)", len(message_content))
        return message_content


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return the configured LLMClient implementation.

    This is the single place that decides which provider is in use. Agents
    should call this factory instead of constructing a provider class directly.
    """
    settings = settings or get_settings()
    config = LLMClientConfig.from_settings(settings)
    return OpenAILLMClient(config)
