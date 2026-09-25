#!/usr/bin/env python3
"""Entry point for the Parakeet Redux STT server.

Run with::

    python server.py
    STT2_PORT=5093 STT2_DEVICE=cpu python server.py

Or directly with uvicorn::

    uvicorn stt2_service.main:app --host 0.0.0.0 --port 5093
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("STT2_HOST", "0.0.0.0")
    port = int(os.getenv("STT2_PORT", "5093"))
    workers = int(os.getenv("STT2_UVICORN_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run(
        "stt2_service.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
