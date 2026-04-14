"""Helpers for categorizing stored model calls."""

from __future__ import annotations

from typing import Any

from sqlalchemy import or_

from app.models import ModelCall

WHISPER_TRANSCRIPTION_PROMPT = "Whisper transcription job"


def whisper_model_call_filter() -> Any:
    """Return a reusable filter that matches transcription model calls."""
    return or_(
        ModelCall.prompt == WHISPER_TRANSCRIPTION_PROMPT,
        ModelCall.model_name.like("%whisper%"),
        ModelCall.model_name.like("local_%"),
    )
