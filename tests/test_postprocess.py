import asyncio

from stt2_service import postprocess


def test_disabled_returns_unchanged(monkeypatch):
    monkeypatch.setattr(postprocess, "POST_PROCESS", False)
    assert asyncio.run(postprocess.postprocess_text("hello")) == "hello"


def test_enabled_without_key_returns_unchanged(monkeypatch):
    monkeypatch.setattr(postprocess, "POST_PROCESS", True)
    monkeypatch.setattr(postprocess, "LITELLM_API_KEY", "")
    monkeypatch.setattr(postprocess, "_client", None)
    assert asyncio.run(postprocess.postprocess_text("hello")) == "hello"


def test_postprocess_many_disabled(monkeypatch):
    monkeypatch.setattr(postprocess, "POST_PROCESS", False)
    assert asyncio.run(postprocess.postprocess_many(["a", "b"])) == ["a", "b"]
