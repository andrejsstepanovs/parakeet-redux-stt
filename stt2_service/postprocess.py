"""Optional LLM post-processing of transcripts via a LiteLLM-compatible proxy.

When enabled (``STT2_POST_PROCESS=true``) the raw ASR text is sent to an
OpenAI-compatible chat endpoint (a LiteLLM proxy) so a small flash model can
clean up punctuation, casing, and obvious transcription errors. If the LLM call
fails or is disabled the original text is returned unchanged, so transcription
never breaks.
"""
from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from .config import (
    LITELLM_API_KEY,
    LITELLM_BASE_URL,
    LITELLM_MODEL,
    POST_PROCESS,
    POST_PROCESS_CONCURRENCY,
    POST_PROCESS_MAX_CHARS,
    POST_PROCESS_SYSTEM_PROMPT,
    POST_PROCESS_TIMEOUT,
    logger,
)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI | None:
    global _client
    if _client is None and LITELLM_API_KEY:
        try:
            _client = AsyncOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                timeout=POST_PROCESS_TIMEOUT,
            )
        except Exception as exc:
            logger.warning(
                "Could not create the post-processing client (%s); keeping raw "
                "transcripts",
                exc,
            )
            return None
    return _client


async def postprocess_text(text: str) -> str:
    """Send one transcript through the LLM; return the original on any failure."""
    if not POST_PROCESS:
        return text
    text = (text or "").strip()
    if not text:
        return text
    client = _get_client()
    if client is None:
        logger.warning(
            "STT2_POST_PROCESS is enabled but LITELLM_API_KEY is not set; "
            "returning raw transcript"
        )
        return text

    if len(text) > POST_PROCESS_MAX_CHARS:
        text = text[:POST_PROCESS_MAX_CHARS]

    try:
        response = await client.chat.completions.create(
            model=LITELLM_MODEL,
            messages=[
                {"role": "system", "content": POST_PROCESS_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=POST_PROCESS_MAX_CHARS,
        )
        corrected = (response.choices[0].message.content or "").strip()
        if not corrected:
            logger.warning("LLM returned empty post-processed text; keeping raw")
            return text
        return corrected
    except Exception as exc:
        logger.warning("LLM post-processing failed (%s); keeping raw transcript", exc)
        return text


async def postprocess_many(texts: list[str]) -> list[str]:
    """Post-process a list of texts with limited concurrency."""
    if not POST_PROCESS:
        return texts
    semaphore = asyncio.Semaphore(POST_PROCESS_CONCURRENCY)

    async def _limited(text: str) -> str:
        async with semaphore:
            try:
                return await postprocess_text(text)
            except Exception as exc:
                # postprocess_text already handles its own failures; this guards
                # against anything unexpected so a request never fails because
                # of the optional cleanup step.
                logger.warning(
                    "Post-processing task failed (%s); keeping raw transcript", exc
                )
                return text

    return list(await asyncio.gather(*(_limited(text) for text in texts)))
