"""E-mail one-time-code (OTP) authentication.

Flow
  1. POST /api/auth/request-code   {email, intent}       a 6-digit code is e-mailed over SMTP
  2. POST /api/auth/verify-code    {challenge_id, code}  returns a single-use, short-lived TICKET
  3. the ticket is exchanged ONCE for a SESSION TOKEN bound to one conversation:
       intent "new"   POST /api/session       creates the conversation for the verified e-mail
       intent "open"  POST /api/auth/open     opens a conversation owned by the verified e-mail
  Every /api/session/{id}/... route then requires that token.

There is no development bypass and no fallback. Without AUTH_SECRET and working SMTP settings the server
refuses to start, and if a code cannot be e-mailed the request fails (nothing is printed to a console instead).

Security properties
  - codes come from `secrets`, are stored only as an HMAC (never in clear), expire after 10 minutes,
    are single use, and lock after 5 wrong guesses
  - requests are rate limited per e-mail address (cooldown + hourly cap) and per client IP
  - tickets and tokens are HMAC-SHA256 signed with AUTH_SECRET; a ticket can be spent exactly once
  - the e-mail a conversation belongs to is fixed when it is created; nobody can open it without a code
    delivered to that mailbox

Accepted .env names (the first one found wins)
  SMTP_HOST      or SMTP_SERVER
  SMTP_FROM      or SMTP_FROM_EMAIL   (if neither is set, SMTP_USERNAME is used as the sender when it is an e-mail address)
"""
import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import time
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path

from . import config  # noqa: F401  (importing it loads .env before AuthSettings.from_env reads the environment)

log = logging.getLogger("tg.auth")

EMAIL_RE = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
INTENTS = ("new", "open")
_ALIASES = {"SMTP_HOST": ("SMTP_SERVER",), "SMTP_FROM": ("SMTP_FROM_EMAIL",)}


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def normalize_email(raw: str) -> str:
    email = (raw or "").strip().lower()
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise AuthError(400, "Please enter a valid email address.")
    return email


@dataclass(frozen=True)
class AuthSettings:
    secret: bytes
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_from: str
    smtp_security: str  # starttls | ssl | none (none is only accepted for a loopback host)
    session_ttl: int = 8 * 3600  # seconds a conversation token stays valid
    otp_ttl: int = 600
    otp_max_attempts: int = 5
    resend_cooldown: int = 60
    max_codes_per_email_hour: int = 5
    max_codes_per_ip_hour: int = 20
    ticket_ttl: int = 600
    smtp_timeout: int = 20

    @classmethod
    def from_env(cls, env=None) -> "AuthSettings":
        env = os.environ if env is None else env

        def get(key: str) -> str:
            for name in (key, *_ALIASES.get(key, ())):
                value = (env.get(name) or "").strip()
                if value:
                    return value
            return ""

        security = (get("SMTP_SECURITY") or "starttls").lower()
        host = get("SMTP_HOST")
        username = get("SMTP_USERNAME")
        # The sender is the mailbox the codes are sent from. When no sender is configured, the SMTP login is used
        # if it is itself an e-mail address (Gmail / Outlook require the sender to be the logged-in account anyway).
        from_raw = get("SMTP_FROM") or (username if EMAIL_RE.match(username) else "")
        local_plain = security == "none" and host.lower() in _LOOPBACK

        missing = [k for k in ("AUTH_SECRET", "SMTP_HOST", "SMTP_PORT") if not get(k)]
        if not from_raw:
            missing.append("SMTP_FROM")
        if not local_plain:
            missing += [k for k in ("SMTP_USERNAME", "SMTP_PASSWORD") if not get(k)]
        if missing:
            raise RuntimeError(
                "Email verification is not configured. Missing in .env: " + ", ".join(missing)
                + ". See .env.example; there is no fallback mode."
            )
        problems = []
        if len(get("AUTH_SECRET")) < 32:
            problems.append("AUTH_SECRET must be at least 32 characters "
                            "(generate one: python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
        if security not in ("starttls", "ssl", "none"):
            problems.append("SMTP_SECURITY must be starttls, ssl or none")
        if security == "none" and host.lower() not in _LOOPBACK:
            problems.append("SMTP_SECURITY=none is only allowed for localhost; use starttls (587) or ssl (465)")
        try:
            port = int(get("SMTP_PORT"))
        except ValueError:
            port = 0
            problems.append("SMTP_PORT must be a number")
        sender = parseaddr(from_raw)[1]
        if not EMAIL_RE.match(sender):
            problems.append("SMTP_FROM must contain a valid address, e.g. TrustGrowth <you@gmail.com>")
        try:
            session_ttl = int(get("AUTH_SESSION_TTL_MIN") or 480) * 60
            if session_ttl <= 0:
                raise ValueError
        except ValueError:
            session_ttl = 0
            problems.append("AUTH_SESSION_TTL_MIN must be a positive number of minutes")
        if problems:
            raise RuntimeError("Invalid email-verification settings: " + "; ".join(problems))
        return cls(
            secret=get("AUTH_SECRET").encode(), smtp_host=host, smtp_port=port,
            smtp_username=username, smtp_password=env.get("SMTP_PASSWORD") or "",
            smtp_from=from_raw, smtp_security=security, session_ttl=session_ttl,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS challenges(
    id TEXT PRIMARY KEY, email TEXT NOT NULL, intent TEXT NOT NULL, code_hash TEXT NOT NULL,
    ip TEXT NOT NULL, created_at REAL NOT NULL, expires_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, done INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_ch_email ON challenges(email, created_at);
CREATE INDEX IF NOT EXISTS ix_ch_ip ON challenges(ip, created_at);
CREATE TABLE IF NOT EXISTS used_tickets(jti TEXT PRIMARY KEY, exp REAL NOT NULL);
"""


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class AuthService:
    def __init__(self, settings: AuthSettings, data_dir):
        self.s = settings
        self.db_path = Path(data_dir) / "auth.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as c:
            c.executescript(_SCHEMA)

    # ------------------------------------------------------------ storage --
    @contextmanager
    def _db(self):
        conn = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)  # autocommit; we BEGIN explicitly
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _tx(self):
        with self._db() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
                c.execute("COMMIT")
            except BaseException:
                if c.in_transaction:
                    c.execute("ROLLBACK")
                raise

    def _code_hash(self, challenge_id: str, code: str) -> str:
        return hmac.new(self.s.secret, f"otp:{challenge_id}:{code}".encode(), hashlib.sha256).hexdigest()

    # -------------------------------------------------- 1. request a code --
    def request_code(self, email: str, intent: str, ip: str) -> dict:
        email = normalize_email(email)
        if intent not in INTENTS:
            raise AuthError(400, "Invalid request.")
        now = time.time()
        challenge_id = secrets.token_urlsafe(18)
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self._tx() as c:
            c.execute("DELETE FROM challenges WHERE created_at < ?", (now - 7200,))
            c.execute("DELETE FROM used_tickets WHERE exp < ?", (now,))
            last = c.execute("SELECT MAX(created_at) FROM challenges WHERE email=?", (email,)).fetchone()[0]
            if last is not None and now - last < self.s.resend_cooldown:
                wait = math.ceil(self.s.resend_cooldown - (now - last))
                raise AuthError(429, f"A code was just sent. Please wait {wait} seconds before requesting another.")
            hour = now - 3600
            n_email = c.execute("SELECT COUNT(*) FROM challenges WHERE email=? AND created_at>?", (email, hour)).fetchone()[0]
            n_ip = c.execute("SELECT COUNT(*) FROM challenges WHERE ip=? AND created_at>?", (ip, hour)).fetchone()[0]
            if n_email >= self.s.max_codes_per_email_hour or n_ip >= self.s.max_codes_per_ip_hour:
                raise AuthError(429, "Too many verification requests. Please try again later.")
            c.execute(
                "INSERT INTO challenges(id,email,intent,code_hash,ip,created_at,expires_at) VALUES(?,?,?,?,?,?,?)",
                (challenge_id, email, intent, self._code_hash(challenge_id, code), ip, now, now + self.s.otp_ttl),
            )
        try:
            self._send(email, code, intent)
        except smtplib.SMTPRecipientsRefused:
            self._forget(challenge_id)
            log.warning("mail server refused recipient %s", email)
            raise AuthError(400, "The mail server rejected that email address. Please check it.")
        except (smtplib.SMTPException, OSError, ssl.SSLError):
            self._forget(challenge_id)
            log.exception("could not send verification email to %s", email)
            raise AuthError(502, "We could not send the verification email. Please try again in a moment.")
        return {"challenge_id": challenge_id, "expires_in": self.s.otp_ttl, "resend_after": self.s.resend_cooldown}

    def _forget(self, challenge_id: str) -> None:
        with self._db() as c:
            c.execute("DELETE FROM challenges WHERE id=?", (challenge_id,))

    def _build_message(self, to: str, code: str, intent: str) -> EmailMessage:
        """The verification e-mail: the code is in the subject (visible without opening the mail), the body says
        why the mail was sent, what to do, and what to do if it was not requested. Plain text plus a simple HTML
        part; ASCII only, so every mail client shows it the same way."""
        purpose = "start a new intake conversation" if intent == "new" else "open an existing intake conversation"
        minutes = self.s.otp_ttl // 60
        msg = EmailMessage()
        msg["Subject"] = f"Your TrustGrowth verification code: {code}"
        msg["From"] = self.s.smtp_from
        msg["To"] = to
        msg["Auto-Submitted"] = "auto-generated"
        msg.set_content(
            f"Hello,\n\n"
            f"Someone entered this email address ({to}) on the TrustGrowth Opportunity Finder to {purpose}. "
            f"To continue, enter this verification code:\n\n"
            f"    {code}\n\n"
            f"The code works once and expires in {minutes} minutes.\n\n"
            f"Did not ask for this? You can ignore this email. Nobody can open your conversations "
            f"without a code sent to this address.\n\n"
            f"--\n"
            f"TrustGrowth Opportunity Finder\n"
            f"This is an automated message. Please do not reply."
        )
        msg.add_alternative(
            '<div style="font-family:Arial,Helvetica,sans-serif;font-size:15px;line-height:1.5;color:#222;max-width:480px">'
            "<p>Hello,</p>"
            f"<p>Someone entered this email address ({to}) on the TrustGrowth Opportunity Finder to {purpose}. "
            "To continue, enter this verification code:</p>"
            f'<p style="font-size:30px;letter-spacing:6px;font-weight:bold;margin:18px 0">{code}</p>'
            f"<p>The code works once and expires in {minutes} minutes.</p>"
            "<p>Did not ask for this? You can ignore this email. Nobody can open your conversations "
            "without a code sent to this address.</p>"
            '<hr style="border:0;border-top:1px solid #ddd;margin:20px 0">'
            '<p style="font-size:12px;color:#777">TrustGrowth Opportunity Finder<br>'
            "This is an automated message. Please do not reply.</p></div>",
            subtype="html",
        )
        return msg

    def _send(self, to: str, code: str, intent: str) -> None:
        msg = self._build_message(to, code, intent)
        ctx = ssl.create_default_context()
        if self.s.smtp_security == "ssl":
            smtp = smtplib.SMTP_SSL(self.s.smtp_host, self.s.smtp_port, timeout=self.s.smtp_timeout, context=ctx)
        else:
            smtp = smtplib.SMTP(self.s.smtp_host, self.s.smtp_port, timeout=self.s.smtp_timeout)
        with smtp:
            smtp.ehlo()
            if self.s.smtp_security == "starttls":
                smtp.starttls(context=ctx)
                smtp.ehlo()
            if self.s.smtp_username:
                smtp.login(self.s.smtp_username, self.s.smtp_password)
            smtp.send_message(msg)

    # --------------------------------------------------- 2. verify a code --
    def verify_code(self, challenge_id: str, code: str) -> dict:
        code = (code or "").strip()
        if not re.fullmatch(r"\d{6}", code) or not challenge_id or len(challenge_id) > 64:
            raise AuthError(400, "Enter the 6-digit code from the email.")
        now = time.time()
        error = None
        with self._tx() as c:
            row = c.execute(
                "SELECT email,intent,code_hash,expires_at,attempts,done FROM challenges WHERE id=?", (challenge_id,)
            ).fetchone()
            if row is None or row[5] or row[3] < now:
                raise AuthError(400, "This code has expired or was already used. Please request a new one.")
            email, intent, stored, _, attempts, _ = row
            if attempts >= self.s.otp_max_attempts:
                raise AuthError(429, "Too many incorrect attempts. Please request a new code.")
            if hmac.compare_digest(stored, self._code_hash(challenge_id, code)):
                c.execute("UPDATE challenges SET done=1 WHERE id=?", (challenge_id,))
            else:
                c.execute("UPDATE challenges SET attempts=attempts+1 WHERE id=?", (challenge_id,))
                left = self.s.otp_max_attempts - attempts - 1
                error = AuthError(400, "That code is not correct. " + (
                    f"{left} attempt{'s' if left != 1 else ''} left." if left > 0 else "Please request a new code."))
        if error:  # raised after the transaction so the failed attempt is really recorded
            raise error
        ticket = self._sign({"typ": "ticket", "email": email, "intent": intent, "jti": secrets.token_urlsafe(16),
                             "exp": now + self.s.ticket_ttl})
        return {"ticket": ticket, "email": email, "intent": intent, "expires_in": self.s.ticket_ttl}

    # ------------------------------------------------------------- tickets --
    def peek_ticket(self, token: str, intent: str) -> str:
        """Validate a ticket without spending it. Returns the verified e-mail."""
        p = self._read(token, "ticket")
        if p.get("intent") != intent:
            raise AuthError(401, "This verification cannot be used for that action. Please verify again.")
        with self._db() as c:
            if c.execute("SELECT 1 FROM used_tickets WHERE jti=?", (p.get("jti"),)).fetchone():
                raise AuthError(401, "This verification was already used. Please verify again.")
        return p["email"]

    def consume_ticket(self, token: str, intent: str) -> str:
        """Spend a ticket. It works exactly once, even under concurrent requests."""
        p = self._read(token, "ticket")
        if p.get("intent") != intent:
            raise AuthError(401, "This verification cannot be used for that action. Please verify again.")
        try:
            with self._db() as c:
                c.execute("INSERT INTO used_tickets(jti, exp) VALUES(?,?)", (p.get("jti"), p["exp"]))
        except sqlite3.IntegrityError:
            raise AuthError(401, "This verification was already used. Please verify again.")
        return p["email"]

    # ------------------------------------------------------ session tokens --
    def issue_session_token(self, email: str, session_id: str) -> str:
        return self._sign({"typ": "session", "email": email, "sid": session_id, "exp": time.time() + self.s.session_ttl})

    def verify_session_token(self, token: str, session_id: str) -> str:
        p = self._read(token, "session")
        if p.get("sid") != session_id:
            raise AuthError(401, "You have not verified access to this conversation. Please verify your email.")
        return p["email"]

    # -------------------------------------------------------------- tokens --
    def _sign(self, payload: dict) -> str:
        body = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
        return f"{body}.{_b64(hmac.new(self.s.secret, body.encode(), hashlib.sha256).digest())}"

    def _read(self, token: str, typ: str) -> dict:
        invalid = AuthError(401, "Verification required. Please verify your email.")
        try:
            body, sig = (token or "").split(".")
            good = _b64(hmac.new(self.s.secret, body.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, good):
                raise invalid
            payload = json.loads(_unb64(body))
        except ValueError:
            raise invalid
        if not isinstance(payload, dict) or payload.get("typ") != typ:
            raise invalid
        if float(payload.get("exp", 0)) < time.time():
            raise AuthError(401, "Your verification has expired. Please verify your email again.")
        return payload
