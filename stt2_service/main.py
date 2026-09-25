"""FastAPI application factory and resource lifespan."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import engine
from .config import DEFAULT_MODEL, logger
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    try:
        logger.info("Lifespan startup: loading default model")
        await asyncio.to_thread(engine.load_model, DEFAULT_MODEL)
        app.state.ready = True
        logger.info("Service ready")
        yield
    finally:
        app.state.ready = False
        logger.info("Lifespan shutdown")
        await asyncio.to_thread(engine.close)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet Redux (ternary) STT",
        version="0.1.0",
        description=(
            "OpenAI-compatible speech-to-text service for Moondream's ternary "
            "Parakeet Redux, running on CPU via Photon."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
