"""Tests for the rate limit on POST /api/predict, through the real app.

The uploads are .txt files on purpose: they are refused with 415 by the
validator without running the model, so each request is cheap and the status
only differs when the limiter steps in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from bacteriacv.model import build_model, save_checkpoint

BAD_FILE = {"file": ("catatan.txt", b"x", "text/plain")}


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    """A throwaway head checkpoint, enough for the app to start."""
    model = build_model(pretrained=False)
    with torch.no_grad():
        model.head_a.weight.zero_()
        model.head_b.weight.zero_()
    return save_checkpoint(model, tmp_path / "heads.pt")


def _client(checkpoint: Path, limit: int | None, monkeypatch, **env: str):
    """Build an app with the given limit and environment."""
    import app.main as app_module

    monkeypatch.delenv("RATE_LIMIT_PER_MINUTE", raising=False)
    monkeypatch.delenv("RATE_LIMIT_TRUST_FORWARDED", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    application = app_module.create_app(checkpoint, rate_limit_per_minute=limit)
    return TestClient(application)


def test_third_predict_call_is_refused_with_429(checkpoint, monkeypatch) -> None:
    """With a limit of 2, calls 1 and 2 pass and call 3 gets 429."""
    with _client(checkpoint, 2, monkeypatch) as client:
        first = client.post("/api/predict", files=BAD_FILE)
        second = client.post("/api/predict", files=BAD_FILE)
        third = client.post("/api/predict", files=BAD_FILE)

    assert first.status_code == 415
    assert second.status_code == 415
    assert third.status_code == 429
    assert third.json() == {"detail": "Too many requests"}
    assert 1 <= int(third.headers["retry-after"]) <= 60


def test_only_predict_is_limited(checkpoint, monkeypatch) -> None:
    """Health and report stay reachable while predict is blocked."""
    with _client(checkpoint, 1, monkeypatch) as client:
        client.post("/api/predict", files=BAD_FILE)
        blocked = client.post("/api/predict", files=BAD_FILE)
        health = client.get("/api/health")
        report = client.get("/api/report")

    assert blocked.status_code == 429
    assert health.status_code == 200
    assert report.status_code in (200, 404)


def test_limit_zero_turns_the_limiter_off(checkpoint, monkeypatch) -> None:
    """With 0, many calls in a row are never refused."""
    with _client(checkpoint, 0, monkeypatch) as client:
        statuses = {client.post("/api/predict", files=BAD_FILE).status_code for _ in range(30)}

    assert statuses == {415}


def test_default_limit_comes_from_the_environment(checkpoint, monkeypatch) -> None:
    """Without an argument, RATE_LIMIT_PER_MINUTE decides."""
    with _client(checkpoint, None, monkeypatch, RATE_LIMIT_PER_MINUTE="1") as client:
        first = client.post("/api/predict", files=BAD_FILE)
        second = client.post("/api/predict", files=BAD_FILE)

    assert first.status_code == 415
    assert second.status_code == 429


def test_forwarded_header_is_ignored_unless_trusted(checkpoint, monkeypatch) -> None:
    """A forged X-Forwarded-For must not give a caller a fresh quota."""
    with _client(checkpoint, 1, monkeypatch) as client:
        first = client.post("/api/predict", files=BAD_FILE, headers={"X-Forwarded-For": "1.1.1.1"})
        second = client.post("/api/predict", files=BAD_FILE, headers={"X-Forwarded-For": "2.2.2.2"})

    assert first.status_code == 415
    assert second.status_code == 429


def test_trusted_forwarded_header_separates_clients(checkpoint, monkeypatch) -> None:
    """With trust on, two different proxy-reported clients have separate quotas."""
    with _client(checkpoint, 1, monkeypatch, RATE_LIMIT_TRUST_FORWARDED="1") as client:
        a_first = client.post("/api/predict", files=BAD_FILE, headers={"X-Forwarded-For": "1.1.1.1"})
        b_first = client.post("/api/predict", files=BAD_FILE, headers={"X-Forwarded-For": "2.2.2.2"})
        a_second = client.post("/api/predict", files=BAD_FILE, headers={"X-Forwarded-For": "1.1.1.1"})

    assert a_first.status_code == 415
    assert b_first.status_code == 415
    assert a_second.status_code == 429


def test_429_carries_cors_headers_for_allowed_origins(checkpoint, monkeypatch) -> None:
    """The site must be able to read the 429, so CORS wraps the limiter."""
    origin = "http://localhost:3000"
    with _client(checkpoint, 1, monkeypatch, CORS_ORIGINS=origin) as client:
        client.post("/api/predict", files=BAD_FILE, headers={"Origin": origin})
        blocked = client.post("/api/predict", files=BAD_FILE, headers={"Origin": origin})

    assert blocked.status_code == 429
    assert blocked.headers["access-control-allow-origin"] == origin
