"""Read chapter metadata from audio files.

MP3 chapters are read from ID3 ``CHAP`` frames; MP4/``M4A`` chapters are read via
ffprobe. The container is sniffed from the file's bytes rather than trusting the
extension, since podcast hosts often serve MP4/AAC audio under a ``.mp3`` name.
"""

import json
import logging
import subprocess
from dataclasses import dataclass

from mutagen.id3 import CHAP, ID3
from mutagen.mp3 import MP3

from podcast_processor.audio_format import detect_audio_container

logger = logging.getLogger("global_logger")


@dataclass
class Chapter:
    """Represents a chapter in an audio file."""

    element_id: str
    title: str
    start_time_ms: int
    end_time_ms: int


def read_chapters(audio_path: str) -> list[Chapter]:
    """
    Read chapters from an audio file, choosing the reader by real container.

    Args:
        audio_path: Path to the audio file

    Returns:
        List of Chapter objects, sorted by start time.
        Returns empty list if no chapters are found or the file cannot be read.
    """
    container = detect_audio_container(audio_path)
    if container == "mp4":
        return _read_chapters_mp4(audio_path)
    return _read_chapters_mp3(audio_path)


def _read_chapters_mp4(audio_path: str) -> list[Chapter]:
    """Read chapters from an MP4/M4A file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-print_format", "json",
                "-show_chapters",
                audio_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        raw_chapters = json.loads(result.stdout).get("chapters", [])
    except (subprocess.CalledProcessError, ValueError) as exc:
        logger.warning("Failed to read chapters from %s: %s", audio_path, exc)
        return []

    chapters: list[Chapter] = []
    for index, entry in enumerate(raw_chapters):
        try:
            start_ms = round(float(entry["start_time"]) * 1000)
            end_ms = round(float(entry["end_time"]) * 1000)
        except (KeyError, ValueError):
            continue
        title = str((entry.get("tags") or {}).get("title") or f"chp{index}")
        chapters.append(
            Chapter(
                element_id=f"chp{index}",
                title=title,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            )
        )

    chapters.sort(key=lambda c: c.start_time_ms)
    logger.info("Found %d chapters in %s", len(chapters), audio_path)
    return chapters


def _read_chapters_mp3(audio_path: str) -> list[Chapter]:
    """Read ID3 CHAP frames from an MP3 file."""
    chapters: list[Chapter] = []

    try:
        audio = MP3(audio_path, ID3=ID3)
        if audio.tags is None:
            logger.debug("No ID3 tags found in %s", audio_path)
            return []

        for _, frame in audio.tags.items():
            if not isinstance(frame, CHAP):
                continue

            element_id = frame.element_id
            start_time_ms = frame.start_time
            end_time_ms = frame.end_time

            # Extract title from sub-frames (TIT2)
            title = ""
            if frame.sub_frames:
                for sub_frame in frame.sub_frames.values():
                    if sub_frame.FrameID == "TIT2":
                        title = str(sub_frame.text[0]) if sub_frame.text else ""
                        break

            if not title:
                title = element_id

            chapters.append(
                Chapter(
                    element_id=element_id,
                    title=title,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )
            )

        # Sort by start time
        chapters.sort(key=lambda c: c.start_time_ms)

        logger.info("Found %d chapters in %s", len(chapters), audio_path)
        for chapter in chapters:
            logger.debug(
                "  Chapter: %s (%d ms - %d ms)",
                chapter.title,
                chapter.start_time_ms,
                chapter.end_time_ms,
            )

    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to read chapters from %s: %s", audio_path, e)
        return []

    return chapters
