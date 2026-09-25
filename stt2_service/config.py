"""Configuration for the Parakeet Redux STT service.

All configuration is read from the environment and validated at import time so
an invalid deployment fails before the Photon runtime downloads or loads any
model weights.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Optional


# Photon/Kestrel runs the ternary GEMMs on a native worker pool. Torch's idle
# OpenMP workers would otherwise spin on the same cores, so this must be set
# before torch is imported anywhere in the process.
os.environ.setdefault("OMP_WAIT_POLICY", "passive")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be >= 1, got {value}")
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric, got {raw!r}") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean, got {raw!r}")


def _env_choice(name: str, default: str, choices: Iterable[str]) -> str:
    allowed = {choice.lower() for choice in choices}
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise RuntimeError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Paths, models and device
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("STT2_MODELS_DIR", ROOT_DIR / "models")).expanduser()
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "true")

# The ternary Parakeet is the CPU-friendly default. ``parakeet-ultra`` is the
# full-precision sibling and only makes sense on CUDA; it is registered too so
# the same service can be pointed at a GPU without code changes.
MODEL_CONFIGS = {
    "moondream/parakeet-redux": {
        "hf_id": "moondream/parakeet-redux",
        "description": "Ternary Parakeet TDT 0.6B v3 (178 MB, CPU-friendly)",
    },
    "moondream/parakeet-ultra": {
        "hf_id": "moondream/parakeet-ultra",
        "description": "Full-precision Parakeet (GPU)",
    },
}

# Accept the OpenAI-style model names clients already use against ~/stt.
MODEL_ALIASES = {
    "parakeet-redux": "moondream/parakeet-redux",
    "parakeet_tdt": "moondream/parakeet-redux",
    "parakeet-tdt-0.6b-v3": "moondream/parakeet-redux",
    "parakeet": "moondream/parakeet-redux",
    "parakeet-ultra": "moondream/parakeet-ultra",
}

DEFAULT_MODEL = os.getenv("STT2_MODEL", "moondream/parakeet-redux").strip().lower()
if DEFAULT_MODEL not in MODEL_CONFIGS:
    raise RuntimeError(
        f"STT2_MODEL must be one of {sorted(MODEL_CONFIGS)}, got {DEFAULT_MODEL!r}"
    )

DEVICE = _env_choice("STT2_DEVICE", "cpu", {"cpu", "cuda", "mps"})

# Optional explicit thread count for the CPU/native worker pool. When unset,
# Kestrel picks one cache domain (physical cores, capped at 8).
CPU_THREADS = _env_optional_int("STT2_CPU_THREADS")

# Parakeet returns native word/character timestamps, so ``word`` is free and
# gives us segments plus per-word timing for every output format.
DEFAULT_TIMESTAMPS = _env_choice(
    "STT2_TIMESTAMPS", "word", {"none", "segment", "word", "character"}
)


# ---------------------------------------------------------------------------
# Request limits and audio handling
# ---------------------------------------------------------------------------
TEMP_DIR = Path(os.getenv("STT2_TEMP_DIR", ROOT_DIR / "temp_uploads")).expanduser()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = _env_int("STT2_MAX_UPLOAD_BYTES", 512 * 1024 * 1024)
MAX_AUDIO_SECONDS = _env_float("STT2_MAX_AUDIO_SECONDS", 2 * 60 * 60)
MAX_BATCH_FILES = _env_int("STT2_MAX_BATCH_FILES", 16)
MAX_BATCH_BYTES = _env_int("STT2_MAX_BATCH_BYTES", 512 * 1024 * 1024)
UPLOAD_READ_CHUNK_BYTES = min(1024 * 1024, MAX_UPLOAD_BYTES)
FFMPEG_TIMEOUT_SEC = _env_float("STT2_FFMPEG_TIMEOUT_SEC", 300.0, minimum=1.0)


# ---------------------------------------------------------------------------
# LLM post-processing (optional GPT cleanup step, shared with ~/stt)
# ---------------------------------------------------------------------------
POST_PROCESS = _env_bool("STT2_POST_PROCESS", False)
DEFAULT_POST_PROCESS_PROMPT = (
    "You are a transcript cleaner. Fix punctuation, capitalization, spacing, "
    "and obvious speech-recognition errors in the transcript below. Keep the "
    "speaker's words and meaning exactly as intended. Do not add, remove, or "
    "rewrite content beyond correcting clear transcription mistakes. Reply "
    "with only the corrected transcript text and nothing else."
)
POST_PROCESS_SYSTEM_PROMPT = os.getenv(
    "STT2_POST_PROCESS_PROMPT", DEFAULT_POST_PROCESS_PROMPT
).strip()
LITELLM_BASE_URL = os.getenv(
    "LITELLM_BASE_URL", "http://127.0.0.1:4000/v1"
).strip().rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "").strip()
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "qwen").strip()
POST_PROCESS_TIMEOUT = _env_float("STT2_POST_PROCESS_TIMEOUT", 90.0, minimum=1.0)
POST_PROCESS_MAX_CHARS = _env_int("STT2_POST_PROCESS_MAX_CHARS", 60_000)
POST_PROCESS_CONCURRENCY = _env_int("STT2_POST_PROCESS_CONCURRENCY", 4)


# ---------------------------------------------------------------------------
# CPU reporting (used by /health and the web UI)
# ---------------------------------------------------------------------------
try:
    _available_logical = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    _available_logical = os.cpu_count() or 1

try:
    import psutil  # type: ignore

    _physical = psutil.cpu_count(logical=False) or _available_logical
except Exception:
    _physical = _available_logical


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("parakeet_redux")

CPU_INFO = {
    "physical": _physical,
    "logical": _available_logical,
    "cpu_threads": CPU_THREADS,
    "device": DEVICE,
}

__all__ = [
    "CPU_INFO",
    "CPU_THREADS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMESTAMPS",
    "DEVICE",
    "FFMPEG_TIMEOUT_SEC",
    "LITELLM_API_KEY",
    "LITELLM_BASE_URL",
    "LITELLM_MODEL",
    "MAX_AUDIO_SECONDS",
    "MAX_BATCH_BYTES",
    "MAX_BATCH_FILES",
    "MAX_UPLOAD_BYTES",
    "MODEL_ALIASES",
    "MODEL_CONFIGS",
    "MODELS_DIR",
    "POST_PROCESS",
    "POST_PROCESS_CONCURRENCY",
    "POST_PROCESS_MAX_CHARS",
    "POST_PROCESS_SYSTEM_PROMPT",
    "POST_PROCESS_TIMEOUT",
    "TEMP_DIR",
    "UPLOAD_READ_CHUNK_BYTES",
    "logger",
]
