"""Tests for e-mail one-time-code verification.

Nothing here is mocked: codes are sent through smtplib to a REAL SMTP server (aiosmtpd, running on 127.0.0.1),
read back out of the delivered message, and submitted through the real FastAPI routes.
Needs the dev packages:  pip install -r requirements-dev.txt
Run:  python -m pytest -q tests/test_auth.py"""
import re
import socket
import sqlite3
import threading
import time
from dataclasses import replace

import pytest
from aiosmtpd.controller import Controller
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tg_intake.auth import AuthError, AuthService, AuthSettings
from tg_intake.engine import IntakeEngine
from tg_intake.store import JsonStore
from tg_intake.web_auth import build_auth


class Inbox:
    def __init__(self):
        self.messages = []

    async def handle_DATA(self, server, session, envelope):
        self.messages.append(envelope)
        return "250 Message accepted"

    def last_code(self, to):
        mails = [m for m in self.messages if to.lower() in m.rcpt_tos]
        assert mails, f"no mail was delivered to {to}"
        return re.search(r"\b(\d{6})\b", mails[-1].content.decode()).group(1)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def inbox():
    box = Inbox()
    port = free_port()
    ctl = Controller(box, hostname="127.0.0.1", port=port)
    ctl.start()
    box.port = port
    yield box
    ctl.stop()


def settings_for(port, **kw):
    return AuthSettings(secret=b"x" * 48, smtp_host="127.0.0.1", smtp_port=port, smtp_username="", smtp_password="",
                        smtp_from="TrustGrowth <noreply@example.com>", smtp_security="none", **kw)


@pytest.fixture
def env(tmp_path, inbox):
    return make_env(tmp_path, settings_for(inbox.port, resend_cooldown=0))


def make_env(tmp_path, settings):
    store = JsonStore(tmp_path)
    auth = AuthService(settings, tmp_path)
    engine = IntakeEngine(None, store)  # new_session never calls the LLM
    view = lambda st: {"session_id": st.session_id, "owner_name": st.owner_name, "phase": st.phase}  # noqa: E731
    router, session_email, load_owned = build_auth(auth, store, engine, view)
    app = FastAPI()
    app.include_router(router)

    @app.get("/api/session/{sid}")
    def get_session(sid: str, email: str = Depends(session_email)):
        return load_owned(sid, email).model_dump()

    class E:
        pass

    e = E()
    e.client, e.auth, e.store, e.path, e.settings = TestClient(app), auth, store, tmp_path, settings
    return e


def H(token):
    return {"Authorization": f"Bearer {token}"}


def verify(env, inbox, email, intent):
    r = env.client.post("/api/auth/request-code", json={"email": email, "intent": intent})
    assert r.status_code == 200, r.text
    code = inbox.last_code(email)
    r = env.client.post("/api/auth/verify-code", json={"challenge_id": r.json()["challenge_id"], "code": code})
    assert r.status_code == 200, r.text
    return r.json()["ticket"]


def create(env, inbox, email, name="Asha"):
    ticket = verify(env, inbox, email, "new")
    r = env.client.post("/api/session", json={"owner_name": name}, headers=H(ticket))
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------ happy paths ---
def test_new_conversation_needs_an_emailed_code_and_binds_the_conversation_to_that_email(env, inbox):
    made = create(env, inbox, "Owner@Example.com")
    assert inbox.messages[-1].rcpt_tos == ["owner@example.com"]  # really delivered over SMTP, normalised
    assert env.store.load(made["session_id"]).owner_email == "owner@example.com"
    r = env.client.get(f"/api/session/{made['session_id']}", headers=H(made["token"]))
    assert r.status_code == 200 and r.json()["owner_email"] == "owner@example.com"


def test_reopening_a_conversation_needs_a_fresh_code_sent_to_its_owner(env, inbox):
    made = create(env, inbox, "owner@example.com")
    sid = made["session_id"]
    assert env.client.get(f"/api/session/{sid}").status_code == 401  # no token, no access

    ticket = verify(env, inbox, "owner@example.com", "open")
    listed = env.client.get("/api/auth/conversations", headers=H(ticket)).json()
    assert [c["id"] for c in listed] == [sid]
    opened = env.client.post("/api/auth/open", json={"session_id": sid}, headers=H(ticket))
    assert opened.status_code == 200
    assert env.client.get(f"/api/session/{sid}", headers=H(opened.json()["token"])).status_code == 200


# --------------------------------------------- other people are kept out ---
def test_someone_else_cannot_open_or_list_a_conversation(env, inbox):
    sid = create(env, inbox, "owner@example.com")["session_id"]
    ticket = verify(env, inbox, "intruder@example.com", "open")
    assert env.client.get("/api/auth/conversations", headers=H(ticket)).json() == []
    r = env.client.post("/api/auth/open", json={"session_id": sid}, headers=H(ticket))
    assert r.status_code == 403


def test_a_token_only_works_for_its_own_conversation(env, inbox):
    a = create(env, inbox, "owner@example.com")
    b = create(env, inbox, "owner@example.com", name="Second")
    assert env.client.get(f"/api/session/{b['session_id']}", headers=H(a["token"])).status_code == 401


@pytest.mark.parametrize("token", ["", "garbage", "a.b", "Bearer x"])
def test_missing_or_bogus_tokens_are_rejected(env, inbox, token):
    sid = create(env, inbox, "owner@example.com")["session_id"]
    headers = H(token) if token else {}
    assert env.client.get(f"/api/session/{sid}", headers=headers).status_code == 401


def test_tampered_token_is_rejected(env, inbox):
    made = create(env, inbox, "owner@example.com")
    body, sig = made["token"].split(".")
    forged = body + "." + ("A" if sig[0] != "A" else "B") + sig[1:]
    assert env.client.get(f"/api/session/{made['session_id']}", headers=H(forged)).status_code == 401


def test_token_signed_with_another_secret_is_rejected(env, inbox, tmp_path):
    made = create(env, inbox, "owner@example.com")
    other = AuthService(replace(env.settings, secret=b"y" * 48), tmp_path / "other")
    forged = other.issue_session_token("owner@example.com", made["session_id"])
    assert env.client.get(f"/api/session/{made['session_id']}", headers=H(forged)).status_code == 401


def test_expired_session_token_is_rejected(tmp_path, inbox):
    e = make_env(tmp_path, settings_for(inbox.port, resend_cooldown=0, session_ttl=-1))
    made = create(e, inbox, "owner@example.com")
    r = e.client.get(f"/api/session/{made['session_id']}", headers=H(made["token"]))
    assert r.status_code == 401 and "expired" in r.json()["detail"]


def test_a_conversation_with_no_verified_owner_is_unreachable(env, inbox):
    from tg_intake.schema import ProjectState
    env.store.save(ProjectState(session_id="abcdef1234", owner_name="legacy"))
    ticket = verify(env, inbox, "owner@example.com", "open")
    assert env.client.post("/api/auth/open", json={"session_id": "abcdef1234"}, headers=H(ticket)).status_code == 403


# ---------------------------------------------------------------- codes ---
def test_wrong_code_is_refused_and_five_wrong_guesses_lock_the_code(env, inbox):
    r = env.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"})
    cid, good = r.json()["challenge_id"], inbox.last_code("o@example.com")
    bad = "000000" if good != "000000" else "111111"
    for i in range(5):
        r = env.client.post("/api/auth/verify-code", json={"challenge_id": cid, "code": bad})
        assert r.status_code == 400 and "not correct" in r.json()["detail"]
    r = env.client.post("/api/auth/verify-code", json={"challenge_id": cid, "code": good})  # even the right one is dead now
    assert r.status_code == 429


def test_a_code_works_once(env, inbox):
    r = env.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"})
    body = {"challenge_id": r.json()["challenge_id"], "code": inbox.last_code("o@example.com")}
    assert env.client.post("/api/auth/verify-code", json=body).status_code == 200
    assert env.client.post("/api/auth/verify-code", json=body).status_code == 400


def test_an_expired_code_is_refused(env, inbox):
    r = env.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"})
    with sqlite3.connect(env.path / "auth.db") as c:
        c.execute("UPDATE challenges SET expires_at = ?", (time.time() - 1,))
    body = {"challenge_id": r.json()["challenge_id"], "code": inbox.last_code("o@example.com")}
    assert env.client.post("/api/auth/verify-code", json=body).status_code == 400


def test_a_code_for_one_challenge_does_not_open_another(env, inbox):
    env.client.post("/api/auth/request-code", json={"email": "a@example.com", "intent": "new"})
    code_a = inbox.last_code("a@example.com")
    r = env.client.post("/api/auth/request-code", json={"email": "b@example.com", "intent": "new"})
    if inbox.last_code("b@example.com") == code_a:
        pytest.skip("the two random codes collided (1 in a million)")
    r = env.client.post("/api/auth/verify-code", json={"challenge_id": r.json()["challenge_id"], "code": code_a})
    assert r.status_code == 400


def test_the_code_is_never_stored_in_clear(env, inbox):
    r = env.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"})
    code, cid = inbox.last_code("o@example.com"), r.json()["challenge_id"]
    with sqlite3.connect(env.path / "auth.db") as c:
        stored = c.execute("SELECT code_hash FROM challenges WHERE id=?", (cid,)).fetchone()[0]
    assert stored != code and stored == env.auth._code_hash(cid, code) and len(stored) == 64


def test_invalid_emails_are_rejected_before_anything_is_sent(env, inbox):
    for bad in ("", "nobody", "a@b", "a b@example.com", "x" * 260 + "@example.com"):
        r = env.client.post("/api/auth/request-code", json={"email": bad, "intent": "new"})
        assert r.status_code == 400
    assert inbox.messages == []


# --------------------------------------------------------------- tickets ---
def test_a_ticket_can_be_spent_only_once(env, inbox):
    ticket = verify(env, inbox, "o@example.com", "new")
    assert env.client.post("/api/session", json={}, headers=H(ticket)).status_code == 200
    assert env.client.post("/api/session", json={}, headers=H(ticket)).status_code == 401


def test_a_ticket_cannot_be_spent_twice_even_concurrently(env, inbox):
    ticket = verify(env, inbox, "o@example.com", "new")
    wins, losses = [], []

    def spend():
        try:
            env.auth.consume_ticket(ticket, "new")
            wins.append(1)
        except AuthError:
            losses.append(1)

    threads = [threading.Thread(target=spend) for _ in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(wins) == 1 and len(losses) == 9


def test_a_ticket_for_one_purpose_cannot_be_used_for_another(env, inbox):
    open_ticket = verify(env, inbox, "o@example.com", "open")
    assert env.client.post("/api/session", json={}, headers=H(open_ticket)).status_code == 401
    new_ticket = verify(env, inbox, "o@example.com", "new")
    assert env.client.post("/api/auth/open", json={"session_id": "abcdef1234"}, headers=H(new_ticket)).status_code == 401


def test_a_session_token_is_not_accepted_as_a_ticket(env, inbox):
    made = create(env, inbox, "o@example.com")
    assert env.client.post("/api/session", json={}, headers=H(made["token"])).status_code == 401


# ---------------------------------------------------------- rate limits ---
def test_resend_cooldown(tmp_path, inbox):
    e = make_env(tmp_path, settings_for(inbox.port, resend_cooldown=60))
    assert e.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"}).status_code == 200
    r = e.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"})
    assert r.status_code == 429 and len(inbox.messages) == 1


def test_hourly_cap_per_email(tmp_path, inbox):
    e = make_env(tmp_path, settings_for(inbox.port, resend_cooldown=0, max_codes_per_email_hour=3))
    codes = [e.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"}).status_code
             for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_hourly_cap_per_ip(tmp_path, inbox):
    e = make_env(tmp_path, settings_for(inbox.port, resend_cooldown=0, max_codes_per_ip_hour=2))
    codes = [e.client.post("/api/auth/request-code", json={"email": f"u{i}@example.com", "intent": "new"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


# ----------------------------------------------- no silent failure modes ---
def test_when_the_mail_cannot_be_sent_the_request_fails_and_no_code_is_left_behind(tmp_path):
    e = make_env(tmp_path, settings_for(free_port(), resend_cooldown=0))  # nothing is listening on that port
    r = e.client.post("/api/auth/request-code", json={"email": "o@example.com", "intent": "new"})
    assert r.status_code == 502
    with sqlite3.connect(tmp_path / "auth.db") as c:
        assert c.execute("SELECT COUNT(*) FROM challenges").fetchone()[0] == 0


def test_startup_refuses_to_run_without_email_settings():
    with pytest.raises(RuntimeError) as err:
        AuthSettings.from_env({})
    text = str(err.value)
    for name in ("AUTH_SECRET", "SMTP_HOST", "SMTP_PORT", "SMTP_FROM", "SMTP_USERNAME", "SMTP_PASSWORD"):
        assert name in text


def test_startup_rejects_weak_secret_and_plaintext_smtp_to_a_remote_host():
    base = {"AUTH_SECRET": "x" * 48, "SMTP_HOST": "smtp.gmail.com", "SMTP_PORT": "587", "SMTP_FROM": "a@example.com",
            "SMTP_USERNAME": "u", "SMTP_PASSWORD": "p"}
    assert AuthSettings.from_env(base).smtp_security == "starttls"
    with pytest.raises(RuntimeError, match="AUTH_SECRET"):
        AuthSettings.from_env({**base, "AUTH_SECRET": "short"})
    with pytest.raises(RuntimeError, match="only allowed for localhost"):
        AuthSettings.from_env({**base, "SMTP_SECURITY": "none"})
    with pytest.raises(RuntimeError, match="SMTP_PORT"):
        AuthSettings.from_env({**base, "SMTP_PORT": "abc"})
