"""Persistent memory. One JSON file per session (full Shared Project State +
transcript + audit log) and, when the owner confirms, a hand-off file for the
next stages. Writes are atomic (temp file, then rename)."""
import json
import logging
import os
import re
from pathlib import Path

from . import framework as fw
from .schema import OpportunityReport, ProjectState, now

log = logging.getLogger("tg.store")
_SID = re.compile(r"[a-f0-9]{6,32}")

VERIFICATION_NOTE = (
    "Every fact below is OWNER-STATED during intake and has NOT been independently verified. "
    "Downstream stages must attach external evidence before treating any of it as verified."
)


def _row(i) -> dict:
    return {
        "id": i.id, "param": i.param, "attribute": i.attribute, "statement": i.statement,
        "origin": i.origin, "owner_confirmed": i.confirmed, "gap_type": i.gap_type or None,
    }


SUGGESTION_NOTE = (
    "System suggestions are paths the intake assistant put forward. They are NOT owner statements and NOT evidence "
    "about the owner. Only 'owner_reaction' and 'owner_reaction_quotes' record what the owner said about them."
)


def _suggestion_row(i) -> dict:
    return {
        "id": i.id, "name": i.attribute, "statement": i.statement, "origin": "system",
        "pros": i.pros, "cons": i.cons, "status": i.status,
        "suggested_by_assistant_as_starting_point": i.recommended,  # the assistant's lean, never the owner's view
        "owner_reaction": i.reaction or None,
        "owner_reaction_quotes": [
            {"turn": e.turn, "quote": e.quote} for e in i.evidence if e.source_type == "owner_statement"
        ],
    }


def build_handoff(st: ProjectState) -> dict:
    act = [i for i in st.active() if i.kind != "suggestion"]  # owner-side items only
    by = lambda kind: [_row(i) for i in act if i.kind == kind]  # noqa: E731
    return {
        "schema": "tg-opportunity-finder/intake-handoff/v2",
        "session_id": st.session_id,
        "owner_name": st.owner_name,
        "mode": st.mode,
        "phase": st.phase,
        "exported_at": now(),
        "verification_note": VERIFICATION_NOTE,
        "shared_project_state": {
            "facts": by("fact"),
            "assumptions": by("assumption"),
            "decisions": by("decision"),
            "alternatives": by("alternative"),
            "information_gaps": by("gap"),
            "evidence": [
                {"item_id": i.id, "source_type": e.source_type, "turn": e.turn, "quote": e.quote}
                for i in act for e in i.evidence
            ],
            "system_suggestions": {
                "note": SUGGESTION_NOTE,
                "paths": [_suggestion_row(i) for i in st.items if i.kind == "suggestion"],
            },
        },
        "parameters": {
            k: {"status": fw.param_status(st, k), "items": [_row(i) for i in act if i.param == k]}
            for k in fw.ORDER + ["constraints", "scope"]
        },
        "owner_questions_for_analysis": [
            q.model_dump() for q in st.owner_questions if q.answer_basis in ("cannot_answer", "general_knowledge_unverified", "")
        ],
    }


class JsonStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "sessions").mkdir(parents=True, exist_ok=True)
        (self.root / "handoff").mkdir(parents=True, exist_ok=True)
        (self.root / "reports").mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        if not _SID.fullmatch(sid):
            raise ValueError("invalid session id")
        return self.root / "sessions" / f"{sid}.json"

    def save(self, st: ProjectState) -> None:
        st.updated_at = now()
        path = self._path(st.session_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(st.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load(self, sid: str) -> ProjectState:
        path = self._path(sid)
        if not path.exists():
            raise FileNotFoundError(f"No session {sid}")
        return ProjectState.model_validate_json(path.read_text(encoding="utf-8"))

    def list_sessions(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "sessions").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            st = ProjectState.model_validate_json(p.read_text(encoding="utf-8"))
            out.append({"id": st.session_id, "owner": st.owner_name, "mode": st.mode, "phase": st.phase, "updated": st.updated_at})
        return out

    def list_for_email(self, email: str) -> list[dict]:
        """Conversations owned by this (already verified) e-mail address, newest first."""
        out = []
        for p in sorted((self.root / "sessions").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("skipping unreadable session file %s", p.name)
                continue
            if str(raw.get("owner_email", "")).strip().lower() != email:
                continue
            st = ProjectState.model_validate(raw)
            out.append({"id": st.session_id, "owner": st.owner_name, "mode": st.mode, "phase": st.phase, "updated": st.updated_at})
        return out

    def export_handoff(self, st: ProjectState) -> Path:
        import json

        path = self.root / "handoff" / f"{st.session_id}.json"
        path.write_text(json.dumps(build_handoff(st), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # -- opportunity evaluation report --------------------------------------
    def _report_path(self, sid: str) -> Path:
        if not _SID.fullmatch(sid):
            raise ValueError("invalid session id")
        return self.root / "reports" / f"{sid}.json"

    def save_report(self, sid: str, report: OpportunityReport) -> Path:
        path = self._report_path(sid)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def load_report(self, sid: str) -> OpportunityReport | None:
        path = self._report_path(sid)
        if not path.exists():
            return None
        return OpportunityReport.model_validate_json(path.read_text(encoding="utf-8"))
