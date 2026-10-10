from typing import Any
from unittest.mock import patch

import httpx
import litellm
import pytest
from pydantic import BaseModel

from shared.llm_completion import INTERACTIONS_URL, llm_completion


class _Schema(BaseModel):
    answer: str


def _interaction(text: str = "hi", status: str = "completed") -> dict[str, Any]:
    return {
        "id": "v1_abc",
        "status": status,
        "steps": [
            {"type": "thought", "signature": "x"},
            {"type": "model_output", "content": [{"type": "text", "text": text}]},
        ],
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 4,
            "total_thought_tokens": 2,
            "total_tokens": 16,
        },
    }


def _response(status: int, body: Any) -> httpx.Response:
    return httpx.Response(
        status, json=body, request=httpx.Request("POST", INTERACTIONS_URL)
    )


def test_non_gemini_models_use_litellm():
    with patch("litellm.completion", return_value="sentinel") as completion:
        result = llm_completion(model="openai/gpt-5-mini", messages=[], temperature=0.1)
    assert result == "sentinel"
    completion.assert_called_once_with(
        model="openai/gpt-5-mini", messages=[], temperature=0.1
    )


def test_gemini_request_shape_and_response_mapping():
    with patch("httpx.post", return_value=_response(200, _interaction())) as post:
        result = llm_completion(
            model="gemini/gemini-3-flash-preview",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": [{"type": "text", "text": "Again"}]},
            ],
            temperature=0.1,
            top_p=0.9,
            max_tokens=300,
            timeout=30,
            api_key="k",
            response_format=_Schema,
        )

    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    assert url == INTERACTIONS_URL
    assert post.call_args.kwargs["headers"] == {"x-goog-api-key": "k"}
    assert body["model"] == "gemini-3-flash-preview"
    assert body["store"] is False
    assert body["system_instruction"] == "Be terse."
    assert [s["type"] for s in body["input"]] == [
        "user_input",
        "model_output",
        "user_input",
    ]
    assert body["input"][2]["content"] == [{"type": "text", "text": "Again"}]
    assert body["generation_config"] == {"max_output_tokens": 300}
    assert body["response_format"]["schema"]["properties"]["answer"]["type"] == "string"
    assert "temperature" not in str(body) and "top_p" not in str(body)

    assert result.choices[0].message.content == "hi"
    assert result.choices[0].finish_reason == "stop"
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 6
    assert result.usage.total_tokens == 16


def test_gemini_json_object_and_max_completion_tokens():
    with patch("httpx.post", return_value=_response(200, _interaction())) as post:
        llm_completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "x"}],
            max_completion_tokens=50,
            response_format={"type": "json_object"},
        )
    body = post.call_args.kwargs["json"]
    assert body["response_format"] == {"type": "text", "mime_type": "application/json"}
    assert "response_mime_type" not in body
    assert body["generation_config"] == {"max_output_tokens": 50}


def test_gemini_incomplete_maps_to_length():
    with patch(
        "httpx.post", return_value=_response(200, _interaction("", "incomplete"))
    ):
        result = llm_completion(model="gemini/gemini-3-flash-preview", messages=[])
    assert result.choices[0].finish_reason == "length"
    assert result.choices[0].message.content is None


@pytest.mark.parametrize(
    "status,body,exc",
    [
        (
            429,
            {"error": {"message": "quota", "code": "resource_exhausted"}},
            litellm.RateLimitError,
        ),
        (
            500,
            {"error": {"message": "boom", "code": "internal"}},
            litellm.InternalServerError,
        ),
        (
            503,
            {"error": {"message": "busy", "code": "unavailable"}},
            litellm.ServiceUnavailableError,
        ),
        (
            404,
            {"error": {"message": "no model", "code": "not_found"}},
            litellm.NotFoundError,
        ),
        (
            400,
            [
                {
                    "error": {
                        "code": 400,
                        "message": "API key not valid",
                        "status": "INVALID_ARGUMENT",
                    }
                }
            ],
            litellm.BadRequestError,
        ),
    ],
)
def test_gemini_errors_raise_litellm_types(status, body, exc):
    with patch("httpx.post", return_value=_response(status, body)):
        with pytest.raises(exc) as info:
            llm_completion(model="gemini/gemini-3-flash-preview", messages=[])
    assert str(status) in str(info.value)


def test_gemini_timeout_raises_litellm_timeout():
    with patch("httpx.post", side_effect=httpx.ReadTimeout("slow")):
        with pytest.raises(litellm.Timeout):
            llm_completion(model="gemini/gemini-3-flash-preview", messages=[])
