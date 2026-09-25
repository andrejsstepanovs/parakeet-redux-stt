import io
import wave

import pytest
from fastapi.testclient import TestClient

from stt2_service import engine
from stt2_service.main import app

FAKE_RESULT = {
    "text": "hello world",
    "language": None,
    "task": "transcribe",
    "duration_seconds": 1.0,
    "source_duration_seconds": 1.0,
    "clip_start_seconds": 0.0,
    "clip_end_seconds": 1.0,
    "segments": [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
        }
    ],
}


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(engine, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(engine, "close", lambda: None)

    async def fake_transcribe(path, model, *, timestamps):
        return FAKE_RESULT

    monkeypatch.setattr(engine, "transcribe", fake_transcribe)
    with TestClient(app) as test_client:
        yield test_client


def _post(client, **data):
    return client.post(
        "/v1/audio/transcriptions",
        data=data,
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
    )


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["ready"] is True
    assert "moondream/parakeet-redux" in body["models"]


def test_transcribe_json(client):
    response = _post(client)
    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}


def test_transcribe_text(client):
    response = _post(client, response_format="text")
    assert response.status_code == 200
    assert response.text == "hello world"


def test_transcribe_srt(client):
    response = _post(client, response_format="srt")
    assert response.status_code == 200
    assert "00:00:00,000 --> 00:00:01,000" in response.text


def test_verbose_json_words(client):
    response = _post(
        client,
        response_format="verbose_json",
        **{"timestamp_granularities[]": "word"},
    )
    body = response.json()
    assert body["duration"] == 1.0
    assert body["words"][0]["word"] == "hello"


def test_unknown_model_rejected(client):
    assert _post(client, model="does-not-exist").status_code == 400


def test_batch(client):
    response = client.post(
        "/v1/audio/transcriptions/batch",
        data={"model": "moondream/parakeet-redux"},
        files=[
            ("files", ("a.wav", _wav_bytes(), "audio/wav")),
            ("files", ("b.wav", _wav_bytes(), "audio/wav")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert [item["text"] for item in body["results"]] == ["hello world", "hello world"]
