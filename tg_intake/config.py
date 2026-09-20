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
    model: str
    api_key: str
    data_dir: Path
    thinking_level: str  # "" = do not send a thinking config (model default)
    history_turns: int


def get_settings() -> Settings:
    return Settings(
        model=os.getenv("TG_MODEL", "gemini-3.1-flash-lite"),
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "",
        data_dir=Path(os.getenv("TG_DATA_DIR", "data")),
        thinking_level=os.getenv("TG_THINKING_LEVEL", ""),
        history_turns=int(os.getenv("TG_HISTORY_TURNS", "8")),
    )
