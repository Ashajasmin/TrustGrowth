"""Gemini wrapper. Two structured-output calls per owner message:
  analyze() -> TurnAnalysis   (what did the owner say?)
  respond() -> ReplyDraft     (what do we say back?)
The model id comes from config (default: gemini-3.1-flash-lite)."""
from typing import Protocol

from pydantic import BaseModel

from .config import Settings
from .prompts import SYSTEM_ANALYZE, SYSTEM_REPLY, SYSTEM_REPORT
from .schema import OpportunityReport, ReplyDraft, TurnAnalysis


class LLMError(Exception):
    pass


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
            return {k: resolve(v) for k, v in node.items() if k != "title"}
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
        self.model = settings.model
        self.thinking_level = settings.thinking_level

    def _call(self, system: str, prompt: str, model_cls: type[BaseModel]):
        types = self._types
        cfg = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_json_schema": json_schema(model_cls),
        }
        if self.thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        last = None
        for _ in range(2):  # one retry on malformed JSON / transient error
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=types.GenerateContentConfig(**cfg)
                )
                return model_cls.model_validate_json(resp.text)
            except Exception as exc:  # noqa: BLE001 - surface as LLMError
                last = exc
        raise LLMError(f"{type(last).__name__}: {last}")

    def analyze(self, prompt: str) -> TurnAnalysis:
        return self._call(SYSTEM_ANALYZE, prompt, TurnAnalysis)

    def respond(self, prompt: str) -> ReplyDraft:
        return self._call(SYSTEM_REPLY, prompt, ReplyDraft)

    def evaluate(self, prompt: str) -> OpportunityReport:
        return self._call(SYSTEM_REPORT, prompt, OpportunityReport)
