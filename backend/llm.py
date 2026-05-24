"""
Thin provider-agnostic LLM client.

Set LLM_PROVIDER=gemini (default) or LLM_PROVIDER=openai in .env.
Uses httpx for both — no heavy SDKs needed.
"""
from __future__ import annotations

import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

# Models
_GEMINI_MODEL = "gemini-2.0-flash-lite"
_OPENAI_MODEL = "gpt-4o-mini"
_GROQ_MODEL   = "llama-3.3-70b-versatile"  # free, fast, generous limits

_TIMEOUT = 60.0


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").lower()


def _call_gemini(system: str, user: str) -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_MODEL}:generateContent?key={key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }
    resp = httpx.post(url, json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(system: str, user: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY', '')}"}
    payload = {
        "model": _GROQ_MODEL,
        "temperature": 0.2,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_openai(system: str, user: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"}
    payload = {
        "model": _OPENAI_MODEL,
        "temperature": 0.2,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = httpx.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


class RateLimitError(Exception):
    """Raised when the LLM API returns a rate-limit response after all retries."""


def chat(system: str, user: str, max_retries: int = 2) -> str:
    """
    Send a prompt and return raw text from whichever provider is active.
    On 429/503 retries once after a short wait, then raises RateLimitError
    so the API can return a clean 429 to the frontend instead of hanging.
    """
    for attempt in range(max_retries):
        try:
            p = _provider()
            if p == "openai":
                return _call_openai(system, user)
            if p == "groq":
                return _call_groq(system, user)
            return _call_gemini(system, user)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 503):
                if attempt < max_retries - 1:
                    time.sleep(5)  # one short retry
                    continue
                provider = _provider()
                raise RateLimitError(
                    f"{provider.title()} API rate limit hit (429). "
                    "Please wait 30–60 seconds and try again, or check your quota/billing."
                ) from exc
            raise


def chat_json(system: str, user: str) -> dict | list:
    """
    Like chat(), but strips markdown fences and parses as JSON.
    Raises ValueError if the response is not valid JSON.
    """
    raw = chat(system, user)
    # strip ```json ... ``` or ``` ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    return json.loads(cleaned)
