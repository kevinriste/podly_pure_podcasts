"""Detect the true audio container of a file from its bytes.

Podcast enclosures frequently carry a misleading URL/extension: several hosts
(e.g. Patreon) advertise ``.mp3`` while actually serving an MP4/AAC (``M4A``)
stream. Trusting the extension causes MP3-only tooling (``mutagen.mp3``, pydub
segment export) to fail with ``HeaderNotFoundError`` deep in the pipeline, so we
sniff the real container from the leading bytes instead.
"""

import logging
from typing import Literal

logger = logging.getLogger("global_logger")

AudioContainer = Literal["mp3", "mp4", "unknown"]

# Map each detected container to the canonical file extension we store it under.
_CONTAINER_EXTENSIONS: dict[AudioContainer, str] = {
    "mp3": ".mp3",
    "mp4": ".m4a",
}


def detect_audio_container(path: str) -> AudioContainer:
    """Return the real container of ``path`` by inspecting its magic bytes.

    Returns ``"mp3"`` for ID3-tagged or raw-MPEG audio, ``"mp4"`` for the
    ISO base-media/MP4 family (``.m4a``/``.m4b``/``.mp4``), and ``"unknown"``
    when neither signature matches.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError as exc:
        logger.warning("Could not read audio header from %s: %s", path, exc)
        return "unknown"

    return _classify_header(head)


def _classify_header(head: bytes) -> AudioContainer:
    # ISO base-media files place a `ftyp` box at bytes 4-8 (e.g. M4A, mp42, isom).
    if head[4:8] == b"ftyp":
        return "mp4"

    # ID3v2-tagged MP3.
    if head[:3] == b"ID3":
        return "mp3"

    # Raw MPEG audio frame sync: 11 set bits (0xFF followed by 0xE0-mask bits).
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"

    return "unknown"


def extension_for_container(container: AudioContainer) -> str | None:
    """Return the canonical extension for a container, or ``None`` if unknown."""
    return _CONTAINER_EXTENSIONS.get(container)
