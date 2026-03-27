"""Tests for the Hunyuan client integration.

These tests verify model/agent construction only — no real API calls are made.
"""

from __future__ import annotations

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from helpers import make_settings
from video_autocut.agent.hunyuan_client import (
    create_agent,
    create_model,
    create_structured_agent,
)

# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------


class TestCreateModel:
    def test_returns_openai_chat_model(self):
        settings = make_settings()
        model = create_model(settings)
        assert isinstance(model, OpenAIChatModel)

    def test_strips_openai_prefix(self):
        settings = make_settings(model_name="openai:hunyuan-turbos-latest")
        model = create_model(settings)
        assert model.model_name == "hunyuan-turbos-latest"

    def test_no_prefix_kept_as_is(self):
        settings = make_settings(model_name="some-custom-model")
        model = create_model(settings)
        assert model.model_name == "some-custom-model"

    def test_provider_uses_custom_base_url(self):
        settings = make_settings(hunyuan_base_url="https://custom.api.com/v1")
        model = create_model(settings)
        # The provider wraps an AsyncOpenAI client; verify it was configured
        client: AsyncOpenAI = model._provider._client
        assert str(client.base_url).rstrip("/").endswith("/v1")

    def test_provider_uses_api_key(self):
        settings = make_settings(hunyuan_api_key="my-secret-key")
        model = create_model(settings)
        client: AsyncOpenAI = model._provider._client
        assert client.api_key == "my-secret-key"

    def test_client_timeout_from_settings(self):
        settings = make_settings(request_timeout_seconds=120)
        model = create_model(settings)
        client: AsyncOpenAI = model._provider._client
        assert client.timeout == 120.0

    def test_client_max_retries_from_settings(self):
        settings = make_settings(max_retries=5)
        model = create_model(settings)
        client: AsyncOpenAI = model._provider._client
        assert client.max_retries == 5

    def test_uses_get_settings_when_none(self, monkeypatch: pytest.MonkeyPatch):
        settings = make_settings(model_name="openai:fallback-model")
        monkeypatch.setattr(
            "video_autocut.agent.hunyuan_client.get_settings",
            lambda: settings,
        )
        model = create_model()
        assert model.model_name == "fallback-model"


# ---------------------------------------------------------------------------
# create_agent
# ---------------------------------------------------------------------------


class TestCreateAgent:
    def test_returns_agent(self):
        settings = make_settings()
        agent = create_agent(settings)
        assert isinstance(agent, Agent)

    def test_default_system_prompt(self):
        settings = make_settings()
        agent = create_agent(settings)
        # PydanticAI stores system prompts as _system_prompts (tuple of callables/strings)
        assert len(agent._system_prompts) > 0

    def test_custom_system_prompt(self):
        settings = make_settings()
        agent = create_agent(settings, system_prompt="You are a test bot.")
        # Verify the custom prompt is stored
        assert any(
            "test bot" in str(p) for p in agent._system_prompts
        )

    def test_retries_from_settings(self):
        settings = make_settings(max_retries=4)
        agent = create_agent(settings)
        assert agent._max_result_retries == 4


# ---------------------------------------------------------------------------
# create_structured_agent
# ---------------------------------------------------------------------------


class _SampleOutput(BaseModel):
    answer: str
    confidence: float


class TestCreateStructuredAgent:
    def test_returns_agent(self):
        settings = make_settings()
        agent = create_structured_agent(_SampleOutput, settings)
        assert isinstance(agent, Agent)

    def test_output_type_is_set(self):
        settings = make_settings()
        agent = create_structured_agent(_SampleOutput, settings)
        # The agent should have output schema configured for _SampleOutput
        assert agent._output_schema is not None

    def test_retries_from_settings(self):
        settings = make_settings(max_retries=7)
        agent = create_structured_agent(_SampleOutput, settings)
        assert agent._max_result_retries == 7


# ---------------------------------------------------------------------------
# video_agent backward compatibility
# ---------------------------------------------------------------------------


class TestVideoAgentCompat:
    def test_create_agent_from_video_agent(self):
        from video_autocut.agent.video_agent import create_agent as compat_create

        settings = make_settings()
        agent = compat_create(settings)
        assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# Re-exports from agent/__init__.py
# ---------------------------------------------------------------------------


class TestAgentInit:
    def test_reexports(self):
        from video_autocut.agent import (
            create_agent,
            create_model,
            create_structured_agent,
        )

        assert callable(create_agent)
        assert callable(create_model)
        assert callable(create_structured_agent)
