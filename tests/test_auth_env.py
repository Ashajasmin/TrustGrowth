"""The .env names used in this project's own env file are accepted, and missing settings still stop the server."""
import pytest

from tg_intake.auth import AuthSettings

SECRET = "s" * 48


def env(**kw):
    return {"AUTH_SECRET": SECRET, "SMTP_PORT": "587", "SMTP_USERNAME": "someone@gmail.com",
            "SMTP_PASSWORD": "app-password", **kw}


def test_smtp_server_and_smtp_from_email_are_accepted_as_alternative_names():
    s = AuthSettings.from_env(env(SMTP_SERVER="smtp.gmail.com", SMTP_FROM_EMAIL="TrustGrowth <sender@gmail.com>"))
    assert s.smtp_host == "smtp.gmail.com" and s.smtp_from == "TrustGrowth <sender@gmail.com>"
    assert s.smtp_security == "starttls" and s.smtp_port == 587


def test_canonical_names_win_over_the_alternatives():
    s = AuthSettings.from_env(env(SMTP_HOST="a.example.com", SMTP_SERVER="b.example.com", SMTP_FROM="a@example.com",
                                  SMTP_FROM_EMAIL="b@example.com"))
    assert s.smtp_host == "a.example.com" and s.smtp_from == "a@example.com"


def test_an_empty_sender_uses_the_smtp_login_when_that_is_an_email_address():
    s = AuthSettings.from_env(env(SMTP_SERVER="smtp.gmail.com", SMTP_FROM_EMAIL=""))
    assert s.smtp_from == "someone@gmail.com"


def test_an_empty_sender_with_a_login_that_is_not_an_email_address_is_an_error():
    with pytest.raises(RuntimeError, match="SMTP_FROM"):
        AuthSettings.from_env(env(SMTP_SERVER="smtp.example.com", SMTP_USERNAME="apikey"))


def test_no_auth_secret_still_stops_the_server():
    e = env(SMTP_SERVER="smtp.gmail.com")
    del e["AUTH_SECRET"]
    with pytest.raises(RuntimeError, match="AUTH_SECRET"):
        AuthSettings.from_env(e)


def test_no_smtp_host_still_stops_the_server():
    with pytest.raises(RuntimeError, match="SMTP_HOST"):
        AuthSettings.from_env(env())
