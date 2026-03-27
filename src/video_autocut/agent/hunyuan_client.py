"""Hunyuan LLM client integration for PydanticAI.

Connects to Tencent Hunyuan (or any OpenAI-compatible endpoint) via
PydanticAI's OpenAI provider.  All connection parameters come from
:mod:`video_autocut.settings`.

Design choices
--------------
* **OpenAI-compatible provider, not a custom one.**
  Hunyuan exposes an OpenAI-compatible REST API, so we reuse PydanticAI's
  built-in ``OpenAIProvider`` + ``OpenAIChatModel`` instead of writing a
  bespoke provider.  Swapping to any other OpenAI-compatible service
  (vLLM, Ollama, Together, LiteLLM …) requires only a settings change.

* **Explicit ``AsyncOpenAI`` client construction.**
  This gives us direct control over ``base_url``, ``api_key``,
  ``max_retries``, and ``timeout`` at the HTTP-transport level, rather than
  relying on environment variables or PydanticAI defaults.

* **Model object vs Agent factory.**
  ``create_model()`` returns a bare ``OpenAIChatModel`` — useful when the
  caller wants to build an ``Agent`` with their own system prompt and
  ``output_type``.  ``create_agent()`` is the higher-level convenience that
  returns a ready-to-use ``Agent[None, str]``.

* **Multimodal readiness.**
  PydanticAI's OpenAI model natively supports ``UserContent`` items that
  include images.  No code change here is needed — callers simply pass
  image content in their ``agent.run()`` messages once the vision pipeline
  is wired up.
"""

from __future__ import annotations

import json as _json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.openai import OpenAIProvider

from video_autocut.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Generic type for structured output models.
OutputT = TypeVar("OutputT", bound=BaseModel)

# ---------------------------------------------------------------------------
# Low-level: model / provider construction
# ---------------------------------------------------------------------------


def create_model(settings: Settings | None = None) -> OpenAIChatModel:
    """Build an ``OpenAIChatModel`` wired to the Hunyuan endpoint.

    Args:
        settings: Explicit settings instance.  When *None*, the cached
            singleton from :func:`get_settings` is used.

    Returns:
        A PydanticAI chat model ready for ``Agent`` construction.
    """
    if settings is None:
        settings = get_settings()

    # The settings.model_name carries the PydanticAI prefix, e.g.
    # "openai:hunyuan-turbos-latest".  We strip the "openai:" prefix when
    # constructing the model object because we supply our own provider.
    model_name = settings.model_name
    if ":" in model_name:
        model_name = model_name.split(":", 1)[1]

    client = AsyncOpenAI(
        base_url=settings.hunyuan_base_url,
        api_key=settings.hunyuan_api_key,
        max_retries=settings.max_retries,
        timeout=float(settings.request_timeout_seconds),
    )

    provider = OpenAIProvider(openai_client=client)
    model = OpenAIChatModel(model_name=model_name, provider=provider)

    logger.info(
        "Hunyuan model ready (model=%s, base_url=%s, retries=%d, timeout=%ds)",
        model_name,
        settings.hunyuan_base_url,
        settings.max_retries,
        settings.request_timeout_seconds,
    )
    return model


# ---------------------------------------------------------------------------
# High-level: agent factories
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a video analysis assistant. You help users analyze video content, "
    "identify key segments, and suggest edits. When asked about a video, provide "
    "structured analysis including scene descriptions, timestamps, and editing "
    "recommendations."
)


def create_agent(
    settings: Settings | None = None,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> Agent[None, str]:
    """Return a text-output agent backed by the Hunyuan model.

    This is the simplest entry point — suitable for freeform Q&A and any
    tool that only needs a plain-text reply.

    Args:
        settings: Explicit settings.  Defaults to :func:`get_settings`.
        system_prompt: System-level instructions for the agent.

    Returns:
        A ``Agent[None, str]`` (no deps, string output).
    """
    model = create_model(settings)
    return Agent(
        model,
        system_prompt=system_prompt,
        retries=settings.max_retries if settings else get_settings().max_retries,
    )


def _build_structured_prompt(system_prompt: str, output_type: type[BaseModel]) -> str:
    """Merge JSON-schema instructions into *system_prompt* so only one system message is sent.

    Hunyuan requires all system messages to appear at the very beginning of the
    message list **and** only accepts a single system message.  PydanticAI's
    ``PromptedOutput`` normally injects a second system message with the schema,
    which Hunyuan rejects.  We work around this by embedding the schema
    instructions directly in the caller's system prompt and passing
    ``template=False`` to suppress the extra message.
    """
    schema = output_type.model_json_schema()
    schema_json = _json.dumps(schema, ensure_ascii=False)
    return (
        f"{system_prompt}\n\n"
        f"Always respond with a JSON object that's compatible with this schema:\n\n"
        f"{schema_json}\n\n"
        f"Don't include any text or Markdown fencing before or after."
    )


def create_structured_agent(
    output_type: type[OutputT],
    settings: Settings | None = None,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> Agent[None, OutputT]:
    """Return an agent that produces structured Pydantic output.

    Use this when you need the LLM to return a typed object — for example
    a ``FrameAnalysis`` or ``ShootingScript``.

    Uses ``PromptedOutput(template=False)`` with JSON-schema instructions
    merged into the system prompt.  This avoids both the ``tool_choice``
    parameter and the extra system message that Hunyuan rejects.

    Args:
        output_type: A Pydantic ``BaseModel`` subclass.  PydanticAI will
            parse and validate the model's text response as JSON.
        settings: Explicit settings.  Defaults to :func:`get_settings`.
        system_prompt: System-level instructions for the agent.

    Returns:
        A typed ``Agent[None, OutputT]``.
    """
    model = create_model(settings)
    merged_prompt = _build_structured_prompt(system_prompt, output_type)
    return Agent(
        model,
        output_type=PromptedOutput(output_type, template=False),
        system_prompt=merged_prompt,
        retries=settings.max_retries if settings else get_settings().max_retries,
    )
