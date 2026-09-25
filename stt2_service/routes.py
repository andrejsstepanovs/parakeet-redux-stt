"""FastAPI routes for the OpenAI-compatible transcription API."""
from __future__ import annotations

import asyncio
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from . import engine
from .audio import convert_to_wav, unlink, write_temp
from .config import (
    CPU_INFO,
    DEFAULT_MODEL,
    DEFAULT_TIMESTAMPS,
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_UPLOAD_BYTES,
    MODEL_CONFIGS,
    POST_PROCESS,
    UPLOAD_READ_CHUNK_BYTES,
    logger,
)
from .postprocess import postprocess_many

router = APIRouter()
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_ALLOWED_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}


# ---------------------------------------------------------------------------
# Text / formatting helpers
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2581", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.replace(" '", "'")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fmt_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(_as_float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    index = 1
    for segment in segments:
        text = (segment.get("segment") or "").strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_fmt_srt_time(segment['start'])} --> {_fmt_srt_time(segment['end'])}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(lines)


def _segments_to_vtt(segments: Sequence[Dict[str, Any]]) -> str:
    output = ["WEBVTT", ""]
    for segment in segments:
        text = (segment.get("segment") or "").strip()
        if not text:
            continue
        start = _fmt_srt_time(segment["start"]).replace(",", ".")
        end = _fmt_srt_time(segment["end"]).replace(",", ".")
        output.extend([f"{start} --> {end}", text, ""])
    return "\n".join(output)


def _result_to_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for raw in result.get("segments") or []:
        if not isinstance(raw, dict):
            continue
        text = _clean_text(raw.get("text") or "")
        if not text:
            continue
        words: List[Dict[str, Any]] = []
        for item in raw.get("words") or []:
            if not isinstance(item, dict):
                continue
            word = (item.get("word") or "").strip()
            if not word:
                continue
            words.append(
                {
                    "word": word,
                    "start": _as_float(item.get("start")),
                    "end": _as_float(item.get("end")),
                }
            )
        segments.append(
            {
                "start": _as_float(raw.get("start")),
                "end": _as_float(raw.get("end")),
                "segment": text,
                "words": words,
            }
        )
    return segments


async def _finalize(
    result: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply optional post-processing and return (text, segments, words)."""
    segments = _result_to_segments(result)
    raw_text = _clean_text(result.get("text") or "")

    if POST_PROCESS and segments:
        corrected = await postprocess_many([item["segment"] for item in segments])
        for item, text in zip(segments, corrected):
            item["segment"] = _clean_text(text)
        full_text = _clean_text(" ".join(item["segment"] for item in segments))
    else:
        full_text = raw_text or _clean_text(
            " ".join(item["segment"] for item in segments)
        )

    words = [word for item in segments for word in item["words"]]
    return full_text, segments, words


# ---------------------------------------------------------------------------
# Validation / IO helpers
# ---------------------------------------------------------------------------
def _ensure_ready(request: Request) -> None:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")


def _validate_model(model: str) -> str:
    try:
        return engine.resolve_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_format(response_format: str) -> str:
    normalized = (response_format or "json").strip().lower()
    if normalized not in _ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format {response_format!r}",
        )
    return normalized


async def _read_upload_limited(upload: UploadFile) -> bytes:
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    declared_size = getattr(upload, "size", None)
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        await upload.close()
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
        )

    payload = bytearray()
    try:
        while True:
            chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                )
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    return bytes(payload)


async def _spool_upload(upload: UploadFile) -> Path:
    raw = await _read_upload_limited(upload)
    suffix = Path(upload.filename or "").suffix.lower()
    return await asyncio.to_thread(write_temp, raw, suffix)


async def _transcribe_path(
    path: Path, model: str, timestamps: str
) -> Dict[str, Any]:
    """Transcribe one file, falling back to an FFmpeg WAV transcode on failure."""
    try:
        return await engine.transcribe(path, model, timestamps=timestamps)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Photon failed on %s (%s); trying ffmpeg", path.name, exc)
        wav = await asyncio.to_thread(convert_to_wav, path)
        if wav is None:
            raise HTTPException(
                status_code=415, detail="Audio could not be decoded"
            ) from exc
        try:
            return await engine.transcribe(wav, model, timestamps=timestamps)
        except Exception as fallback_exc:
            logger.exception("Transcription failed after ffmpeg fallback")
            raise HTTPException(
                status_code=415, detail="Audio could not be decoded"
            ) from fallback_exc
        finally:
            await asyncio.to_thread(unlink, wav)


# ---------------------------------------------------------------------------
# Web UI / ops endpoints
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = _TEMPLATES_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="UI not available")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@router.get("/parakeet.png")
def logo():
    path = _TEMPLATES_DIR.parent / "parakeet.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type="image/png")


@router.get("/health")
def health(request: Request):
    ready = bool(getattr(request.app.state, "ready", False))
    return {
        "status": "healthy" if ready else "starting",
        "ready": ready,
        "models": list(MODEL_CONFIGS.keys()),
        "loaded": engine.loaded_models(),
        "default_model": DEFAULT_MODEL,
        "default_timestamps": DEFAULT_TIMESTAMPS,
        "post_process": POST_PROCESS,
        "cpu": CPU_INFO,
    }


@router.get("/healthz")
def healthz(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


@router.get("/metrics")
def metrics():
    try:
        import psutil  # type: ignore
    except Exception:
        raise HTTPException(status_code=501, detail="psutil is not available")
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": memory.percent,
        "ram_used_gb": round(memory.used / (1024**3), 2),
        "ram_total_gb": round(memory.total / (1024**3), 2),
    }


# ---------------------------------------------------------------------------
# Transcription endpoints
# ---------------------------------------------------------------------------
@router.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    response_format: str = Form("json"),
    timestamp_granularities: Optional[List[str]] = Form(
        None, alias="timestamp_granularities[]"
    ),
    timestamp_granularities_plain: Optional[List[str]] = Form(
        None, alias="timestamp_granularities"
    ),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
):
    del language, prompt, temperature  # accepted for OpenAI client compatibility
    _ensure_ready(request)
    model_name = _validate_model(model)
    output_format = _validate_format(response_format)

    path = await _spool_upload(file)
    started = time.perf_counter()
    try:
        result = await _transcribe_path(
            path, model_name, timestamps=DEFAULT_TIMESTAMPS
        )
    finally:
        await asyncio.to_thread(unlink, path)

    full_text, segments, words = await _finalize(result)
    duration = _as_float(
        result.get("duration_seconds") or result.get("source_duration_seconds")
    )

    logger.info(
        "transcribe model=%s dur=%.2fs segments=%d elapsed=%.0fms",
        model_name,
        duration,
        len(segments),
        (time.perf_counter() - started) * 1000,
    )

    if output_format == "text":
        return PlainTextResponse(full_text)
    if output_format == "srt":
        return Response(_segments_to_srt(segments), media_type="application/x-subrip")
    if output_format == "vtt":
        return Response(_segments_to_vtt(segments), media_type="text/vtt")
    if output_format == "verbose_json":
        granularities = set(timestamp_granularities or []) | set(
            timestamp_granularities_plain or []
        )
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result.get("language") or "auto",
                "duration": duration,
                "text": full_text,
                "segments": [
                    {
                        "id": index,
                        "seek": 0,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["segment"],
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                    for index, segment in enumerate(segments)
                ],
                "words": words if "word" in granularities else None,
            }
        )
    return JSONResponse({"text": full_text})


@router.post("/v1/audio/transcriptions/batch")
async def transcribe_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    model: str = Form(DEFAULT_MODEL),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains {len(files)} files; limit is {MAX_BATCH_FILES}",
        )
    _ensure_ready(request)
    model_name = _validate_model(model)
    filenames = [upload.filename or "unnamed" for upload in files]

    paths: List[Path] = []
    total_bytes = 0
    try:
        for upload in files:
            path = await _spool_upload(upload)
            total_bytes += path.stat().st_size
            if total_bytes > MAX_BATCH_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Batch exceeds the {MAX_BATCH_BYTES} byte limit",
                )
            paths.append(path)

        raw_results = await asyncio.gather(
            *(
                _transcribe_path(path, model_name, DEFAULT_TIMESTAMPS)
                for path in paths
            ),
            return_exceptions=True,
        )
    finally:
        for path in paths:
            await asyncio.to_thread(unlink, path)

    response_items = []
    for filename, item in zip(filenames, raw_results):
        if isinstance(item, BaseException):
            if isinstance(item, HTTPException):
                raise item
            logger.exception(
                "batch transcription failed for %s",
                filename,
                exc_info=(type(item), item, item.__traceback__),
            )
            raise HTTPException(
                status_code=415, detail=f"{filename}: audio could not be decoded"
            )
        text, _segments, _words = await _finalize(item)
        response_items.append(
            {
                "filename": filename,
                "text": text,
                "duration": _as_float(
                    item.get("duration_seconds")
                    or item.get("source_duration_seconds")
                ),
            }
        )

    return {"results": response_items, "batch_size": len(response_items)}
