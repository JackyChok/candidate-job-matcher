"""
Tests for the LLM client wrapper — focused on what we can test without
hitting the real network: JSON parsing, markdown stripping, retry logic.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from backend.llm import RateLimitError, chat, chat_json


# ---------------------------------------------------------------------------
# chat_json: markdown fence stripping
# ---------------------------------------------------------------------------


def test_chat_json_strips_json_code_fence():
    fake_response = '```json\n{"key": "value"}\n```'
    with patch("backend.llm.chat", return_value=fake_response):
        result = chat_json("sys", "user")
    assert result == {"key": "value"}


def test_chat_json_strips_plain_code_fence():
    fake_response = '```\n{"key": "value"}\n```'
    with patch("backend.llm.chat", return_value=fake_response):
        result = chat_json("sys", "user")
    assert result == {"key": "value"}


def test_chat_json_handles_no_fence():
    fake_response = '{"key": "value"}'
    with patch("backend.llm.chat", return_value=fake_response):
        result = chat_json("sys", "user")
    assert result == {"key": "value"}


def test_chat_json_parses_lists():
    fake_response = '[{"a": 1}, {"a": 2}]'
    with patch("backend.llm.chat", return_value=fake_response):
        result = chat_json("sys", "user")
    assert isinstance(result, list)
    assert len(result) == 2


def test_chat_json_raises_on_invalid_json():
    with patch("backend.llm.chat", return_value="not json at all"):
        with pytest.raises(json.JSONDecodeError):
            chat_json("sys", "user")


# ---------------------------------------------------------------------------
# chat: retry + rate limit handling
# ---------------------------------------------------------------------------


def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    """Build a synthetic httpx.HTTPStatusError with the given status."""
    request = httpx.Request("POST", "https://example.com")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("err", request=request, response=response)


def test_chat_retries_on_429_then_raises_rate_limit():
    """Two 429s in a row should produce a RateLimitError after the retry budget."""
    with patch("backend.llm._provider", return_value="gemini"), \
         patch("backend.llm._call_gemini", side_effect=_make_http_error(429)), \
         patch("backend.llm.time.sleep"):  # avoid real waits
        with pytest.raises(RateLimitError):
            chat("sys", "user")


def test_chat_succeeds_after_one_retry():
    """If the first call 429s but the second succeeds, return the result."""
    call_count = {"n": 0}

    def flaky(_sys, _user):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _make_http_error(429)
        return "success"

    with patch("backend.llm._provider", return_value="gemini"), \
         patch("backend.llm._call_gemini", side_effect=flaky), \
         patch("backend.llm.time.sleep"):
        result = chat("sys", "user")
    assert result == "success"
    assert call_count["n"] == 2


def test_chat_does_not_retry_on_400():
    """Non-429/503 errors should bubble up immediately."""
    with patch("backend.llm._provider", return_value="gemini"), \
         patch("backend.llm._call_gemini", side_effect=_make_http_error(400)):
        with pytest.raises(httpx.HTTPStatusError):
            chat("sys", "user")


def test_chat_provider_dispatch_openai():
    with patch("backend.llm._provider", return_value="openai"), \
         patch("backend.llm._call_openai", return_value="openai ok") as mock_openai, \
         patch("backend.llm._call_gemini") as mock_gemini:
        result = chat("sys", "user")
    assert result == "openai ok"
    mock_openai.assert_called_once()
    mock_gemini.assert_not_called()


def test_chat_provider_dispatch_groq():
    with patch("backend.llm._provider", return_value="groq"), \
         patch("backend.llm._call_groq", return_value="groq ok") as mock_groq:
        result = chat("sys", "user")
    assert result == "groq ok"
    mock_groq.assert_called_once()


def test_chat_provider_dispatch_gemini_default():
    with patch("backend.llm._provider", return_value="gemini"), \
         patch("backend.llm._call_gemini", return_value="gemini ok") as mock_gemini:
        result = chat("sys", "user")
    assert result == "gemini ok"
    mock_gemini.assert_called_once()
