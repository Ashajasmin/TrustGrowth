"""Web API + chat page.   uvicorn server:app --reload   ->  http://127.0.0.1:8000"""
import threading
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tg_intake import framework as fw
from tg_intake.config import get_settings
from tg_intake.engine import IntakeEngine
from tg_intake.llm import GeminiLLM, LLMError
from tg_intake.report import generate_report
from tg_intake.store import JsonStore, build_handoff

settings = get_settings()
store = JsonStore(settings.data_dir)
engine = IntakeEngine(GeminiLLM(settings), store, settings.history_turns)
locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
app = FastAPI(title="TG Opportunity Finder - Intake")


class NewSession(BaseModel):
    owner_name: str = ""


class Message(BaseModel):
    text: str


class ModeBody(BaseModel):
    mode: str


def view(st) -> dict:
    return {
        "session_id": st.session_id,
        "owner_name": st.owner_name,
        "mode": st.mode,
        "phase": st.phase,
        "asking": st.last_focus,  # topic the bot just asked about
        "coverage": [{"key": k, "label": fw.label_of(k), "status": s} for k, s in fw.coverage(st).items()],
        "items": [
            {
                "id": i.id, "kind": i.kind, "param": i.param, "gap_type": i.gap_type,
                "origin": i.origin, "confirmed": i.confirmed, "statement": i.statement,
            }
            for i in st.active()
        ],
    }


def _load(sid: str):
    try:
        return store.load(sid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Session not found")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/sessions")
def sessions():
    return store.list_sessions()


@app.post("/api/session")
def new_session(body: NewSession):
    st = engine.new_session(body.owner_name)
    return {"reply": st.transcript[-1].text, **view(st)}


@app.get("/api/session/{sid}")
def get_session(sid: str):
    st = _load(sid)
    return {"transcript": [t.model_dump() for t in st.transcript], **view(st)}


@app.post("/api/session/{sid}/message")
def message(sid: str, body: Message):
    with locks[sid]:
        st = _load(sid)
        result = engine.handle(st, body.text)
        return {"reply": result.reply, "complete": result.complete, **view(result.state)}


@app.post("/api/session/{sid}/mode")
def set_mode(sid: str, body: ModeBody):
    if body.mode not in fw.MODES:
        raise HTTPException(400, "mode must be brief, balanced or detailed")
    with locks[sid]:
        st = _load(sid)
        st.mode = body.mode
        st.audit.append({"turn": st.turn, "event": "mode_change_ui", "to": body.mode})
        store.save(st)
        return view(st)


@app.get("/api/session/{sid}/handoff")
def handoff(sid: str):
    return build_handoff(_load(sid))


@app.get("/api/session/{sid}/report")
def get_report(sid: str):
    _load(sid)  # 404 if the session itself does not exist
    report = store.load_report(sid)
    if report is None:
        raise HTTPException(404, "No report generated yet for this session.")
    return report.model_dump()


@app.post("/api/session/{sid}/report")
def create_report(sid: str):
    st = _load(sid)
    if st.phase != "complete":
        raise HTTPException(400, "Confirm the intake before generating the opportunity evaluation report.")
    with locks[sid]:
        handoff_data = build_handoff(st)
        try:
            report = generate_report(handoff_data, engine.llm)
        except LLMError as exc:
            raise HTTPException(502, f"Report generation failed: {exc}")
        store.save_report(sid, report)
        return report.model_dump()
