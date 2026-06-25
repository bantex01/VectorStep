from fastapi import HTTPException
import pytest

import src.main as main


def test_no_tokens_configured_skips_auth_entirely(monkeypatch):
    monkeypatch.setattr(main, "_webhook_tokens", {})

    assert main._resolve_team("") is None
    assert main._resolve_team("Bearer anything") is None


def test_legacy_single_token_resolves_to_no_team(monkeypatch):
    monkeypatch.setattr(main, "_webhook_tokens", {"secret": None})

    assert main._resolve_team("Bearer secret") is None


def test_unknown_token_raises_401(monkeypatch):
    monkeypatch.setattr(main, "_webhook_tokens", {"secret": None})

    with pytest.raises(HTTPException) as exc_info:
        main._resolve_team("Bearer wrong")
    assert exc_info.value.status_code == 401


def test_missing_auth_header_raises_401_when_tokens_configured(monkeypatch):
    monkeypatch.setattr(main, "_webhook_tokens", {"secret": None})

    with pytest.raises(HTTPException) as exc_info:
        main._resolve_team("")
    assert exc_info.value.status_code == 401


def test_team_token_resolves_to_team_name(monkeypatch):
    monkeypatch.setattr(
        main, "_webhook_tokens", {"tok-a": "payments", "tok-b": "platform"}
    )

    assert main._resolve_team("Bearer tok-a") == "payments"
    assert main._resolve_team("Bearer tok-b") == "platform"


def test_non_bearer_header_raises_401(monkeypatch):
    monkeypatch.setattr(main, "_webhook_tokens", {"secret": None})

    with pytest.raises(HTTPException) as exc_info:
        main._resolve_team("secret")
    assert exc_info.value.status_code == 401
