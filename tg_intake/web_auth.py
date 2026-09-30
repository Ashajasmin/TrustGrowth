"""HTTP layer for e-mail verification: the /api/auth/* routes, the POST /api/session route (which now needs a
verified ticket) and the dependency that guards every /api/session/{sid}/... route."""
from typing import Callable, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from .auth import AuthError, AuthService


class CodeRequest(BaseModel):
    email: str
    intent: Literal["new", "open"]


class CodeCheck(BaseModel):
    challenge_id: str
    code: str


class OpenBody(BaseModel):
    session_id: str


class NewSession(BaseModel):
    owner_name: str = ""


def bearer(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer ") or not authorization[7:].strip():
        raise HTTPException(401, "Verification required. Please verify your email.")
    return authorization[7:].strip()


def _http(exc: AuthError) -> HTTPException:
    return HTTPException(exc.status, exc.message)


def build_auth(auth: AuthService, store, engine, view: Callable):
    """Returns (router, session_email, load_owned).
    session_email : dependency for /api/session/{sid}/... routes; yields the verified e-mail or raises 401
    load_owned    : loads a conversation and checks it belongs to that e-mail (404 / 403 otherwise)"""
    router = APIRouter()

    def session_email(sid: str, token: str = Depends(bearer)) -> str:
        try:
            return auth.verify_session_token(token, sid)
        except AuthError as exc:
            raise _http(exc)

    def load_owned(sid: str, email: str):
        try:
            st = store.load(sid)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "Session not found")
        if not st.owner_email or st.owner_email != email:
            raise HTTPException(403, "This conversation belongs to a different verified email.")
        return st

    @router.post("/api/auth/request-code")
    def request_code(body: CodeRequest, request: Request):
        ip = request.client.host if request.client else "unknown"
        try:
            return auth.request_code(body.email, body.intent, ip)
        except AuthError as exc:
            raise _http(exc)

    @router.post("/api/auth/verify-code")
    def verify_code(body: CodeCheck):
        try:
            return auth.verify_code(body.challenge_id, body.code)
        except AuthError as exc:
            raise _http(exc)

    @router.get("/api/auth/conversations")
    def my_conversations(ticket: str = Depends(bearer)):
        """Conversations owned by the e-mail that was just verified (ticket is NOT spent here)."""
        try:
            email = auth.peek_ticket(ticket, "open")
        except AuthError as exc:
            raise _http(exc)
        return store.list_for_email(email)

    @router.post("/api/auth/open")
    def open_conversation(body: OpenBody, ticket: str = Depends(bearer)):
        try:
            email = auth.peek_ticket(ticket, "open")
            st = load_owned(body.session_id, email)
            auth.consume_ticket(ticket, "open")
            return {"session_id": st.session_id, "token": auth.issue_session_token(email, st.session_id)}
        except AuthError as exc:
            raise _http(exc)

    @router.post("/api/session")
    def new_session(body: NewSession, ticket: str = Depends(bearer)):
        try:
            email = auth.consume_ticket(ticket, "new")
        except AuthError as exc:
            raise _http(exc)
        st = engine.new_session(body.owner_name, owner_email=email)
        return {"reply": st.transcript[-1].text, "token": auth.issue_session_token(email, st.session_id), **view(st)}

    return router, session_email, load_owned
