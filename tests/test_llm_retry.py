"""Retry / fallback behaviour of the Gemini wrapper, with a scripted stand-in for the API client (no network).
Run:  python -m pytest -q tests/test_llm_retry.py"""
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tg_intake.llm import GeminiLLM, LLMError, is_transient


class Out(BaseModel):
    x: int


class ApiError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FakeClient:
    """`script` holds one entry per call: an Exception to raise, or the JSON text to return."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


def make(script, fallbacks=("backup",), attempts=3):
    llm = GeminiLLM.__new__(GeminiLLM)  # skip __init__: it needs the real SDK and an API key
    llm._types = SimpleNamespace(
        ThinkingConfig=lambda **kw: kw, GenerateContentConfig=lambda **kw: kw)
    llm._client = FakeClient(script)
    llm.models = {"analyze": "main", "reply": "main", "report": "main"}
    llm.thinking_level = "high"
    llm.fallback_models = list(fallbacks)
    llm.attempts_per_model = attempts
    llm.base_delay = 0
    llm.sleeps = []
    llm._sleep = llm.sleeps.append
    return llm


BUSY = ApiError(503, "503 UNAVAILABLE. This model is currently experiencing high demand.")


def test_a_busy_model_is_retried_and_then_succeeds():
    llm = make([BUSY, BUSY, '{"x": 1}'])
    assert llm._call("analyze", "sys", "hi", Out).x == 1
    assert llm._client.calls == ["main", "main", "main"]
    assert len(llm.sleeps) == 2


def test_a_model_that_stays_busy_hands_over_to_the_fallback_model():
    llm = make([BUSY, BUSY, BUSY, '{"x": 2}'])
    assert llm._call("analyze", "sys", "hi", Out).x == 2
    assert llm._client.calls == ["main", "main", "main", "backup"]


def test_malformed_json_is_retried():
    llm = make(["not json at all", '{"x": 3}'])
    assert llm._call("reply", "sys", "hi", Out).x == 3


def test_a_non_retryable_error_skips_straight_to_the_next_model():
    llm = make([ApiError(404, "model not found"), '{"x": 4}'])
    assert llm._call("analyze", "sys", "hi", Out).x == 4
    assert llm._client.calls == ["main", "backup"]


def test_llmerror_only_when_every_model_has_failed():
    llm = make([BUSY] * 6)
    with pytest.raises(LLMError, match="UNAVAILABLE"):
        llm._call("analyze", "sys", "hi", Out)
    assert llm._client.calls == ["main"] * 3 + ["backup"] * 3


def test_thinking_level_is_only_sent_to_the_primary_model():
    llm = make([])
    assert llm._config("sys", Out, "main", "main").get("thinking_config") == {"thinking_level": "HIGH"}
    assert "thinking_config" not in llm._config("sys", Out, "backup", "main")


def test_transient_classification():
    assert is_transient(ApiError(503, "x")) and is_transient(ApiError(429, "x"))
    assert not is_transient(ApiError(400, "bad request")) and not is_transient(ApiError(403, "key"))
    assert is_transient(TimeoutError("read timed out"))
