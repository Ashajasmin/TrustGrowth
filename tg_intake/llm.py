"""Gemini wrapper. Two structured-output calls per owner message:
  analyze() -> TurnAnalysis   (what did the owner say?)
  respond() -> ReplyDraft     (what do we say back?)
A third call, evaluate(), writes the report.
Each role has its own model id in config (TG_MODEL_ANALYZE / TG_MODEL_REPLY / TG_MODEL_REPORT),
falling back to TG_MODEL (default: gemini-3.8-flash)."""
import logging
import random
import time
from typing import Protocol

from pydantic import BaseModel

from .config import Settings
from .prompts import SYSTEM_ANALYZE, SYSTEM_REPLY, SYSTEM_REPORT
from .schema import OpportunityReport, OpportunityReportOut, ReplyDraft, TurnAnalysis


log = logging.getLogger("tg.llm")


class LLMError(Exception):
    pass


# Errors that are worth trying again: Google's servers are busy (503 "high demand"), rate limited (429), a gateway
# or timeout problem, a dropped connection, or the model returned JSON that did not fit the schema.
_TRANSIENT_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_WORDS = ("unavailable", "overloaded", "high demand", "resource_exhausted", "deadline", "timed out",
                    "timeout", "temporarily", "connection", "reset by peer", "internal", "try again")


def is_transient(exc: Exception) -> bool:
    """True when trying the same request again (after a short wait) can succeed."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        if code in _TRANSIENT_CODES:
            return True
        if 400 <= code < 500:  # bad request, bad key, unknown model: waiting will not help
            return False
    name = type(exc).__name__.lower()
    if "validation" in name or "json" in name:  # malformed / off-schema model output: a new sample usually fixes it
        return True
    text = f"{name} {exc}".lower()
    return any(w in text for w in _TRANSIENT_WORDS)


class LLM(Protocol):
    def analyze(self, prompt: str) -> TurnAnalysis: ...
    def respond(self, prompt: str) -> ReplyDraft: ...
    def evaluate(self, prompt: str) -> OpportunityReport: ...


def json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema with $ref/$defs inlined and titles dropped (simpler for the API)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


class GeminiLLM:
    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise LLMError("No API key found. Put GEMINI_API_KEY=... in the .env file.")
        from google import genai  # imported here so tests do not need the SDK
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=settings.api_key)
        self.models = {
            "analyze": settings.model_analyze,
            "reply": settings.model_reply,
            "report": settings.model_report,
        }
        self.thinking_level = settings.thinking_level  # one level for every role, if set
        self.thinking_levels = {
            "analyze": settings.thinking_analyze,
            "reply": settings.thinking_reply,
            "report": settings.thinking_report,
        }
        self.fallback_models = list(settings.fallback_models)
        self.attempts_per_model = max(1, settings.llm_attempts)
        self.base_delay = 1.5  # seconds; doubles after every failed attempt (plus a little jitter)
        self._sleep = time.sleep

    def _chain(self, role: str) -> list:
        """The role's own model first, then the fallbacks, without repeats."""
        chain = []
        for m in [self.models[role], *self.fallback_models]:
            if m and m not in chain:
                chain.append(m)
        return chain

    def verify_models(self) -> list:
        """Ask the Gemini API whether each configured primary model id exists. Returns one message per id that Google
        says does not exist (404). Network trouble or a busy service is NOT reported here: retries handle that."""
        problems, seen = [], set()
        for role, model in self.models.items():
            if model in seen:
                continue
            seen.add(model)
            try:
                self._client.models.get(model=model)
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code == 404:
                    problems.append(f"model '{model}' (used for: {', '.join(r for r, m in self.models.items() if m == model)}) "
                                    "does not exist in the Gemini API. Check the exact id on ai.google.dev/gemini-api/docs/models "
                                    "(for example the Pro model is 'gemini-3.1-pro-preview', not 'gemini-3.1-pro').")
                else:
                    log.warning("could not verify model '%s' (%s): %s", model, role, str(exc)[:200])
        return problems

    def _config(self, system: str, model_cls: type[BaseModel], model: str, primary: str, role: str = ""):
        types = self._types
        cfg = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_json_schema": json_schema(model_cls),
        }
        # The thinking level was chosen for the primary model; fallback models use their own default.
        level = (getattr(self, "thinking_levels", {}).get(role) or self.thinking_level or "") if role else (self.thinking_level or "")
        if level and model == primary:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=level.upper())
        return types.GenerateContentConfig(**cfg)

    def _call(self, role: str, system: str, prompt: str, model_cls: type[BaseModel]):
        """Ask Gemini. A busy or rate-limited model is retried with growing waits; if it stays unavailable the next
        model in the chain is tried. Only when every model has failed does the caller see an LLMError."""
        chain = self._chain(role)
        errors = []
        for model in chain:
            for attempt in range(1, self.attempts_per_model + 1):
                try:
                    resp = self._client.models.generate_content(
                        model=model, contents=prompt, config=self._config(system, model_cls, model, chain[0], role)
                    )
                    return model_cls.model_validate_json(resp.text)
                except Exception as exc:  # noqa: BLE001 - classified below, surfaced as LLMError if nothing works
                    errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:200]}")
                    transient = is_transient(exc)
                    log.warning("Gemini call failed (%s, attempt %d/%d, %s): %s", model, attempt,
                                self.attempts_per_model, "will retry" if transient else "not retryable", exc)
                    if not transient:
                        break  # same request would fail again: move on to the next model
                    if attempt < self.attempts_per_model:
                        self._sleep(self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
        raise LLMError(" | ".join(errors[-4:]) or "no model configured")

    def analyze(self, prompt: str) -> TurnAnalysis:
        return self._call("analyze", SYSTEM_ANALYZE, prompt, TurnAnalysis)

    def respond(self, prompt: str) -> ReplyDraft:
        return self._call("reply", SYSTEM_REPLY, prompt, ReplyDraft)

    def evaluate(self, prompt: str) -> OpportunityReport:
        return self._call("report", SYSTEM_REPORT, prompt, OpportunityReportOut)
