import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


@dataclass(frozen=True)
class Settings:
    model: str  # default model, used for any role without its own setting
    model_analyze: str  # extraction of what the owner said (structured, quote-exact)
    model_reply: str  # the conversation agent: reflects, suggests paths, talks to the owner
    model_report: str  # the evaluation report
    api_key: str
    data_dir: Path
    thinking_level: str  # "" = do not send a thinking config (model default)
    history_turns: int
    fallback_models: tuple = ()  # tried in order when the chosen model stays busy / unavailable
    llm_attempts: int = 4  # tries per model (waits 1.5s, 3s, 6s in between) before moving to the next model


def _csv(value: str) -> tuple:
    return tuple(x.strip() for x in (value or "").split(",") if x.strip())


def get_settings() -> Settings:
    model = os.getenv("TG_MODEL", "gemini-3.8-flash")
    return Settings(
        model=model,
        model_analyze=os.getenv("TG_MODEL_ANALYZE") or model,
        model_reply=os.getenv("TG_MODEL_REPLY") or model,
        model_report=os.getenv("TG_MODEL_REPORT") or model,
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        data_dir=Path(os.getenv("TG_DATA_DIR", "data")),
        thinking_level=os.getenv("TG_THINKING_LEVEL", ""),
        history_turns=int(os.getenv("TG_HISTORY_TURNS", "8")),
        fallback_models=_csv(os.getenv("TG_FALLBACK_MODELS", "gemini-3.8-flash,gemini-flash-latest")),
        llm_attempts=max(1, int(os.getenv("TG_LLM_ATTEMPTS", "4"))),
    )
