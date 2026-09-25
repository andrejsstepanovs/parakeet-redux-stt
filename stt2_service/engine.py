"""Thread-safe Photon/Kestrel client management for Parakeet Redux.

The `moondream` client owns a background event loop and an inference engine.
We keep one cached client per model id for the process lifetime so the engine
can micro-batch concurrent transcriptions. Models are loaded lazily: the
default is warmed at startup and others are loaded on first request.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    CPU_THREADS,
    DEFAULT_MODEL,
    DEVICE,
    MODEL_ALIASES,
    MODEL_CONFIGS,
    logger,
)

_CLIENT_LOCK = threading.RLock()
_CLIENTS: Dict[str, Any] = {}


def resolve_model(name: Optional[str]) -> str:
    """Normalize a requested model name to a registered Photon model id."""
    normalized = (name or DEFAULT_MODEL).strip().lower()
    normalized = MODEL_ALIASES.get(normalized, normalized)
    if normalized not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {sorted(MODEL_CONFIGS)}"
        )
    return normalized


def _open_client(model_name: str):
    import moondream as md

    kwargs: Dict[str, Any] = {"device": DEVICE}
    if CPU_THREADS is not None:
        kwargs["cpu_threads"] = CPU_THREADS
    logger.info(
        "Loading Photon model %s (device=%s, cpu_threads=%s)",
        model_name,
        DEVICE,
        CPU_THREADS,
    )
    return md.photon(model_name, **kwargs)


def _safe_close(client: Any) -> None:
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        logger.warning("Failed to close Photon client cleanly: %s", exc)


def load_model(name: str = DEFAULT_MODEL):
    """Load (or return the cached) Photon client for ``name``."""
    normalized = resolve_model(name)
    with _CLIENT_LOCK:
        cached = _CLIENTS.get(normalized)
        if cached is not None:
            return cached
        client = _open_client(normalized)
        _CLIENTS[normalized] = client
        logger.info("Loaded %s", normalized)
        return client


def get_client(name: str = DEFAULT_MODEL):
    return load_model(name)


def loaded_models() -> List[str]:
    with _CLIENT_LOCK:
        return sorted(_CLIENTS)


def close() -> None:
    with _CLIENT_LOCK:
        for client in _CLIENTS.values():
            _safe_close(client)
        _CLIENTS.clear()


async def transcribe(path: Path, model: str, *, timestamps: str) -> Dict[str, Any]:
    """Transcribe an encoded audio file and return the raw Photon result."""
    client = get_client(model)
    result = await client.atranscribe(audio=Path(path), timestamps=timestamps)
    if not isinstance(result, dict):
        raise RuntimeError("Photon returned a non-final transcription result")
    return result
