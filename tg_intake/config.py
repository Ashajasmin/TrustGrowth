import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# Thinking levels accepted by the Gemini 3 series models used here (Google's "Gemini thinking" page, checked Sep 2026):
# low / medium / high. "minimal" is NOT accepted by gemini-3.8-flash and makes the call fail, so it is refused up front.
THINKING_LEVELS = ("", "low", "medium", "high")

# Defaults per role (override with TG_THINKING_LEVEL_ANALYZE / _REPLY / _REPORT, or one level for all with TG_THINKING_LEVEL):
#   analyze - extraction of what the owner said: structured and quote-exact, runs on every turn -> low keeps it quick
#   reply   - the conversation itself, where the quality of the whole product is decided        -> high
#   report  - the evaluation report (written after intake, not tuned yet)                        -> high
ROLE_THINKING_DEFAULTS = {"analyze": "low", "reply": "high", "report": "high"}


@dataclass(frozen=True)
class Settings:
    model: str  # default model, used for any role without its own setting
    model_analyze: str  # extraction of what the owner said (structured, quote-exact)
    model_reply: str  # the conversation agent: reflects, suggests paths, talks to the owner
    model_report: str  # the evaluation report
    api_key: str
    data_dir: Path
    thinking_level: str  # one level for every role ("" = use the per-role defaults below)
    history_turns: int
    fallback_models: tuple = ()  # tried in order when the chosen model stays busy / unavailable
    llm_attempts: int = 4  # tries per model (waits 1.5s, 3s, 6s in between) before moving to the next model
    thinking_analyze: str = "low"
    thinking_reply: str = "high"
    thinking_report: str = "high"


def _csv(value: str) -> tuple:
    return tuple(x.strip() for x in (value or "").split(",") if x.strip())


def _level(name: str, fallback: str) -> str:
    """Role-specific env var, else the single TG_THINKING_LEVEL, else the role's default."""
    value = (os.getenv(name) or os.getenv("TG_THINKING_LEVEL") or fallback).strip().lower()
    if value not in THINKING_LEVELS:
        raise ValueError(f"{name}/TG_THINKING_LEVEL must be one of low, medium, high (got '{value}'). "
                         "'minimal' is not supported by gemini-3.8-flash.")
    return value


def get_settings() -> Settings:
    model = os.getenv("TG_MODEL", "gemini-3.8-flash")
    return Settings(
        model=model,
        model_analyze=os.getenv("TG_MODEL_ANALYZE") or model,
        model_reply=os.getenv("TG_MODEL_REPLY") or model,
        model_report=os.getenv("TG_MODEL_REPORT") or model,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        data_dir=Path(os.getenv("TG_DATA_DIR", "data")),
        thinking_level=_level("TG_THINKING_LEVEL", ""),
        history_turns=int(os.getenv("TG_HISTORY_TURNS", "8")),
        fallback_models=_csv(os.getenv("TG_FALLBACK_MODELS", "gemini-3.7-flash,gemini-3.5-flash")),
        llm_attempts=max(1, int(os.getenv("TG_LLM_ATTEMPTS", "4"))),
        thinking_analyze=_level("TG_THINKING_LEVEL_ANALYZE", ROLE_THINKING_DEFAULTS["analyze"]),
        thinking_reply=_level("TG_THINKING_LEVEL_REPLY", ROLE_THINKING_DEFAULTS["reply"]),
        thinking_report=_level("TG_THINKING_LEVEL_REPORT", ROLE_THINKING_DEFAULTS["report"]),
    )
