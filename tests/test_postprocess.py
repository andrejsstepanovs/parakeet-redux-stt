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


class _FailingCompletions:
    async def create(self, **kwargs):
        raise RuntimeError("connection refused")


class _FailingClient:
    class chat:
        completions = _FailingCompletions()


def test_api_failure_falls_back_to_raw(monkeypatch):
    monkeypatch.setattr(postprocess, "POST_PROCESS", True)
    monkeypatch.setattr(postprocess, "LITELLM_API_KEY", "sk-test")
    monkeypatch.setattr(postprocess, "_client", _FailingClient())
    assert asyncio.run(postprocess.postprocess_text("hello")) == "hello"


def test_postprocess_many_survives_task_exception(monkeypatch):
    monkeypatch.setattr(postprocess, "POST_PROCESS", True)

    async def boom(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(postprocess, "postprocess_text", boom)
    assert asyncio.run(postprocess.postprocess_many(["a", "b"])) == ["a", "b"]
