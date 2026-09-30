"""Web API + chat page.   uvicorn server:app --reload   ->  http://127.0.0.1:8000

Every conversation route requires e-mail verification (see tg_intake/auth.py). The server does not start
unless AUTH_SECRET and the SMTP_* settings are present in .env."""
import threading
from collections import defaultdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tg_intake import framework as fw
from tg_intake.auth import AuthService, AuthSettings
from tg_intake.config import get_settings
from tg_intake.engine import IntakeEngine
from tg_intake.llm import GeminiLLM, LLMError
from tg_intake.report import generate_report
from tg_intake.store import JsonStore, build_handoff
from tg_intake.web_auth import build_auth

settings = get_settings()
store = JsonStore(settings.data_dir)
engine = IntakeEngine(GeminiLLM(settings), store, settings.history_turns)
locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
app = FastAPI(title="TG Opportunity Finder - Intake")

try:
    auth = AuthService(AuthSettings.from_env(), settings.data_dir)
except RuntimeError as exc:
    raise SystemExit(f"\nCannot start: {exc}\n")


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
        "asking": st.last_focus,  # topic the bot just asked about ("" if its last reply asked nothing)
        "paths_offered": st.path_rounds,
        "coverage": [{"key": k, "label": fw.label_of(k), "status": s} for k, s in fw.coverage(st).items()],
        "items": [
            {
                "id": i.id, "kind": i.kind, "param": i.param, "gap_type": i.gap_type,
                "origin": i.origin, "confirmed": i.confirmed, "statement": i.statement,
                "reaction": i.reaction, "pros": i.pros, "cons": i.cons,
            }
            for i in st.active()
        ],
    }


# /api/auth/*  and  POST /api/session (needs a verified ticket)
router, session_email, load_owned = build_auth(auth, store, engine, view)
app.include_router(router)


@app.get("/")
def index():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-store"},  # always serve the latest UI, never a cached copy
    )


@app.get("/api/session/{sid}")
def get_session(sid: str, email: str = Depends(session_email)):
    st = load_owned(sid, email)
    return {"transcript": [t.model_dump() for t in st.transcript], **view(st)}


@app.post("/api/session/{sid}/message")
def message(sid: str, body: Message, email: str = Depends(session_email)):
    with locks[sid]:
        st = load_owned(sid, email)
        result = engine.handle(st, body.text)
        return {"reply": result.reply, "complete": result.complete, **view(result.state)}


@app.post("/api/session/{sid}/mode")
def set_mode(sid: str, body: ModeBody, email: str = Depends(session_email)):
    if body.mode not in fw.MODES:
        raise HTTPException(400, "mode must be brief, balanced or detailed")
    with locks[sid]:
        st = load_owned(sid, email)
        st.mode = body.mode
        st.audit.append({"turn": st.turn, "event": "mode_change_ui", "to": body.mode})
        store.save(st)
        return view(st)


@app.get("/api/session/{sid}/handoff")
def handoff(sid: str, email: str = Depends(session_email)):
    return build_handoff(load_owned(sid, email))


@app.get("/api/session/{sid}/report")
def get_report(sid: str, email: str = Depends(session_email)):
    load_owned(sid, email)  # 404 if the session does not exist, 403 if it is not this e-mail's
    report = store.load_report(sid)
    if report is None:
        raise HTTPException(404, "No report generated yet for this session.")
    return report.model_dump()


@app.post("/api/session/{sid}/report")
def create_report(sid: str, email: str = Depends(session_email)):
    st = load_owned(sid, email)
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
