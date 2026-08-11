"""Unit tests for the provider-independent LLM client abstraction.

None of these tests contact a real LLM provider: the OpenAI SDK's chat
completions call is mocked at the client boundary, so the whole suite
passes even when no real LLM_API_KEY is configured.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import openai
import pytest

from core.config import Settings
from core.llm_client import (
    LLMAuthenticationError,
    LLMClientConfig,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    OpenAILLMClient,
    get_llm_client,
)


def make_client(api_key: str = "test-key-123") -> OpenAILLMClient:
    config = LLMClientConfig(
        api_key=api_key,
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )
    return OpenAILLMClient(config)


def fake_completion(content: str) -> SimpleNamespace:
    """Build a minimal object shaped like an OpenAI ChatCompletion response."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_generate_returns_content_on_success():
    client = make_client()
    client._client.chat.completions.create = MagicMock(
        return_value=fake_completion("Structured test case JSON here")
    )

    result = client.generate("Summarize this requirement")

    assert result == "Structured test case JSON here"


def test_generate_rejects_empty_prompt():
    client = make_client()
    with pytest.raises(ValueError):
        client.generate("   ")


def test_generate_raises_configuration_error_when_api_key_missing():
    client = make_client(api_key="")

    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        client.generate("Any prompt")


def test_generate_raises_authentication_error_on_invalid_key():
    client = make_client()
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=401, request=request)
    client._client.chat.completions.create = MagicMock(
        side_effect=openai.AuthenticationError("Invalid API key", response=response, body=None)
    )

    with pytest.raises(LLMAuthenticationError):
        client.generate("Any prompt")


def test_generate_raises_request_error_on_connection_failure():
    client = make_client()
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    client._client.chat.completions.create = MagicMock(
        side_effect=openai.APIConnectionError(request=request)
    )

    with pytest.raises(LLMRequestError):
        client.generate("Any prompt")


def test_generate_raises_response_error_on_empty_content():
    client = make_client()
    client._client.chat.completions.create = MagicMock(return_value=fake_completion(""))

    with pytest.raises(LLMResponseError):
        client.generate("Any prompt")


def test_generate_raises_response_error_on_malformed_response():
    client = make_client()
    client._client.chat.completions.create = MagicMock(return_value=SimpleNamespace(choices=[]))

    with pytest.raises(LLMResponseError):
        client.generate("Any prompt")


def test_api_key_never_appears_in_config_repr_or_str():
    config = LLMClientConfig(
        api_key="super-secret-value",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
    )
    assert "super-secret-value" not in repr(config)
    assert "super-secret-value" not in str(config)


def test_factory_builds_openai_client_from_settings():
    settings = Settings(
        llm_api_key="test-key",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
    )
    client = get_llm_client(settings)
    assert isinstance(client, OpenAILLMClient)
