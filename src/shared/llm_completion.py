"""Single entry point for chat completions.

Gemini models (``gemini/...``) go through Google's Interactions API; generateContent
(what litellm.completion uses for Gemini) is legacy, and newer Gemini models reject
temperature/top_p/top_k. Everything else is passed to litellm.completion unchanged.

The Gemini path returns a litellm ``ModelResponse`` and raises litellm exception
types, so callers and the retry/error classification code need no Gemini special cases.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import litellm
from litellm import ModelResponse
from pydantic import BaseModel

INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
GEMINI_PREFIX = "gemini/"


def is_gemini_model(model: str | None) -> bool:
    return bool(model) and str(model).startswith(GEMINI_PREFIX)


def llm_completion(**kwargs: Any) -> Any:
    if not is_gemini_model(kwargs.get("model")):
        return litellm.completion(**kwargs)
    return _gemini_interaction(**kwargs)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI-style content parts
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


# Schema-less JSON mode. A bare response_mime_type is rejected by Interactions
# ("responseFormat must be set when responseMimeType is set").
_JSON_MODE = {"type": "text", "mime_type": "application/json"}


def _response_format(response_format: Any) -> dict[str, Any]:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return {
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_format.model_json_schema(),
            }
        }
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            return {"response_format": _JSON_MODE}
        if response_format.get("type") == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            if schema:
                return {
                    "response_format": {
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": schema,
                    }
                }
            return {"response_format": _JSON_MODE}
    return {}


def _build_request(kwargs: dict[str, Any]) -> dict[str, Any]:
    """temperature/top_p/top_k/penalties are deliberately not forwarded: deprecated
    for Gemini (newer models return 400), and Interactions has no penalties."""
    model = str(kwargs["model"])[len(GEMINI_PREFIX) :]
    system_parts: list[str] = []
    steps: list[dict[str, Any]] = []
    for message in kwargs.get("messages") or []:
        role = message.get("role")
        text = _text(message.get("content"))
        if role in ("system", "developer"):
            system_parts.append(text)
            continue
        steps.append(
            {
                "type": "model_output" if role == "assistant" else "user_input",
                "content": [{"type": "text", "text": text}],
            }
        )

    request: dict[str, Any] = {"model": model, "store": False, "input": steps}
    if system_parts:
        request["system_instruction"] = "\n\n".join(system_parts)

    generation_config: dict[str, Any] = {}
    max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
    if max_tokens:
        generation_config["max_output_tokens"] = int(max_tokens)
    if generation_config:
        request["generation_config"] = generation_config

    request.update(_response_format(kwargs.get("response_format")))
    return request


def _api_key(kwargs: dict[str, Any]) -> str | None:
    return (
        kwargs.get("api_key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )


def _raise_for_status(response: httpx.Response, model: str) -> None:
    if response.is_success:
        return
    try:
        body = response.json()
        error = body[0] if isinstance(body, list) and body else body
        error = error.get("error", error) if isinstance(error, dict) else {}
        message = error.get("message") or response.text
    except ValueError:
        message = response.text
    message = f"Gemini Interactions API error {response.status_code}: {message}"
    status = response.status_code
    args = {"message": message, "llm_provider": "gemini", "model": model}
    if status == 429:
        raise litellm.RateLimitError(response=response, **args)
    if status in (401, 403):
        raise litellm.AuthenticationError(response=response, **args)
    if status == 404:
        raise litellm.NotFoundError(response=response, **args)
    if status == 503:
        raise litellm.ServiceUnavailableError(response=response, **args)
    if status >= 500:
        raise litellm.InternalServerError(response=response, **args)
    raise litellm.BadRequestError(response=response, **args)


def _to_model_response(interaction: dict[str, Any], model: str) -> ModelResponse:
    text = "".join(
        content.get("text", "")
        for step in interaction.get("steps") or []
        if step.get("type") == "model_output"
        for content in step.get("content") or []
        if content.get("type") == "text"
    )
    usage = interaction.get("usage") or {}
    output_tokens = (usage.get("total_output_tokens") or 0) + (
        usage.get("total_thought_tokens") or 0
    )
    return ModelResponse(
        id=interaction.get("id"),
        model=model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": text or None},
                # "incomplete" = a token budget (max_output_tokens) was exhausted.
                "finish_reason": (
                    "length" if interaction.get("status") == "incomplete" else "stop"
                ),
            }
        ],
        usage={
            "prompt_tokens": usage.get("total_input_tokens") or 0,
            "completion_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens") or 0,
        },
    )


def _gemini_interaction(**kwargs: Any) -> ModelResponse:
    model = str(kwargs["model"])
    request = _build_request(kwargs)
    timeout = kwargs.get("timeout") or 600
    try:
        response = httpx.post(
            INTERACTIONS_URL,
            headers={"x-goog-api-key": _api_key(kwargs) or ""},
            json=request,
            timeout=float(timeout),
        )
    except httpx.TimeoutException as exc:
        raise litellm.Timeout(
            message=f"Gemini Interactions API timeout: {exc}",
            model=model,
            llm_provider="gemini",
        ) from exc
    except httpx.HTTPError as exc:
        raise litellm.APIConnectionError(
            message=f"Gemini Interactions API connection error: {exc}",
            llm_provider="gemini",
            model=model,
        ) from exc
    _raise_for_status(response, model)
    return _to_model_response(response.json(), model)
