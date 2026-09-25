"""Audio input handling: spill uploads to disk and an FFmpeg fallback."""
from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from .config import FFMPEG_TIMEOUT_SEC, TEMP_DIR, logger


def write_temp(data: bytes, suffix: str = "") -> Path:
    """Write uploaded bytes to a unique file in the temp directory."""
    safe_suffix = (
        suffix if suffix.startswith(".") and 1 < len(suffix) <= 12 else ".bin"
    )
    fd, name = tempfile.mkstemp(prefix="stt2_", suffix=safe_suffix, dir=str(TEMP_DIR))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
    except Exception:
        unlink(name)
        raise
    return Path(name)


def unlink(path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def convert_to_wav(src) -> Optional[Path]:
    """Best-effort FFmpeg conversion to 16 kHz mono PCM WAV.

    Photon already decodes the common containers itself; this is only used as a
    fallback for an input it cannot decode.
    """
    src = Path(src)
    dst = TEMP_DIR / f"stt2_{uuid.uuid4().hex}.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(dst),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        logger.warning("ffmpeg is not installed; cannot fall back for %s", src.name)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg conversion timed out for %s", src.name)
        return None
    if process.returncode != 0 or not dst.exists():
        logger.warning(
            "ffmpeg conversion failed for %s: %s",
            src.name,
            process.stderr.decode(errors="replace").strip()[:300],
        )
        unlink(dst)
        return None
    return dst
