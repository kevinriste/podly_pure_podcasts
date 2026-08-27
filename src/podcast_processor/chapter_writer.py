"""Write chapter metadata to processed audio files with adjusted timestamps.

Chapters are embedded with the writer appropriate to the file's real container:
ID3 ``CHAP``/``CTOC`` frames for MP3, and (via ffmpeg's ``FFMETADATA``) QuickTime
chapters for the MP4/``M4A`` family. The container is sniffed from the file's
bytes rather than trusting its extension, because podcast hosts routinely serve
MP4/AAC audio under a ``.mp3`` name.
"""

import logging
import os
import subprocess
import tempfile
from typing import cast

from mutagen.id3 import CHAP, CTOC, ID3, TIT2
from mutagen.mp3 import MP3

from podcast_processor.audio_format import detect_audio_container
from podcast_processor.chapter_reader import Chapter

logger = logging.getLogger("global_logger")


def recalculate_chapter_times(
    chapters: list[Chapter],
    removed_segments: list[tuple[float, float]],
) -> list[Chapter]:
    """
    Adjust chapter timestamps after ad segment removal.

    For each chapter, subtract the cumulative duration of all
    removed segments that came before it.

    Args:
        chapters: List of chapters to adjust
        removed_segments: List of (start_sec, end_sec) tuples that were removed

    Returns:
        New list of Chapter objects with adjusted timestamps
    """
    if not chapters:
        return []

    if not removed_segments:
        return chapters

    # Normalize removed segments to sorted, non-overlapping millisecond windows so
    # offset math always reflects unique removed audio time before each marker.
    sorted_segments_ms = _normalize_removed_segments_ms(removed_segments)

    adjusted_chapters: list[Chapter] = []

    for chapter in chapters:
        chapter_start_ms = chapter.start_time_ms
        chapter_end_ms = chapter.end_time_ms
        start_offset_ms = _removed_offset_ms_at_time(
            chapter_start_ms, sorted_segments_ms
        )
        end_offset_ms = _removed_offset_ms_at_time(chapter_end_ms, sorted_segments_ms)

        end_offset_ms = max(end_offset_ms, start_offset_ms)

        # Apply offsets independently so cuts inside a chapter shrink its duration.
        new_start_ms = max(0, chapter_start_ms - start_offset_ms)
        new_end_ms = max(new_start_ms, chapter_end_ms - end_offset_ms)

        adjusted_chapters.append(
            Chapter(
                element_id=chapter.element_id,
                title=chapter.title,
                start_time_ms=new_start_ms,
                end_time_ms=new_end_ms,
            )
        )

        logger.debug(
            "Adjusted chapter '%s': %d ms -> %d ms (offset: %d ms)",
            chapter.title,
            chapter_start_ms,
            new_start_ms,
            start_offset_ms,
        )

    return adjusted_chapters


def _normalize_removed_segments_ms(
    removed_segments: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Convert to sorted, merged millisecond windows."""
    windows_ms: list[tuple[int, int]] = []
    for start_sec, end_sec in removed_segments:
        start_ms = round(start_sec * 1000)
        end_ms = round(end_sec * 1000)
        if end_ms <= start_ms:
            continue
        windows_ms.append((start_ms, end_ms))

    if not windows_ms:
        return []

    windows_ms.sort(key=lambda window: window[0])
    merged: list[tuple[int, int]] = [windows_ms[0]]

    for start_ms, end_ms in windows_ms[1:]:
        last_start_ms, last_end_ms = merged[-1]
        if start_ms <= last_end_ms:
            merged[-1] = (last_start_ms, max(last_end_ms, end_ms))
            continue
        merged.append((start_ms, end_ms))

    return merged


def _removed_offset_ms_at_time(
    time_ms: int,
    sorted_segments_ms: list[tuple[int, int]],
) -> int:
    """Return cumulative removed audio before a given original timestamp."""
    offset_ms = 0
    for seg_start_ms, seg_end_ms in sorted_segments_ms:
        if seg_end_ms <= time_ms:
            offset_ms += max(0, seg_end_ms - seg_start_ms)
            continue
        if seg_start_ms < time_ms:
            offset_ms += max(0, time_ms - seg_start_ms)
        break
    return offset_ms


def write_chapters(
    audio_path: str,
    chapters: list[Chapter],
) -> None:
    """
    Write chapter metadata to an audio file, overwriting any existing chapters.

    The correct embedding is chosen from the file's real container (sniffed from
    its bytes), so MP4/``M4A`` audio served under a ``.mp3`` name is handled
    instead of crashing the MP3-only path.

    Args:
        audio_path: Path to the audio file
        chapters: List of Chapter objects to write

    Raises:
        ChapterWriteError: If the container is unsupported or embedding fails.
    """
    if not chapters:
        logger.info("No chapters to write to %s", audio_path)
        return

    # Sort chapters by start time to ensure correct order
    sorted_chapters = sorted(chapters, key=lambda c: c.start_time_ms)

    container = detect_audio_container(audio_path)
    if container == "mp3":
        _write_chapters_mp3(audio_path, sorted_chapters)
    elif container == "mp4":
        _write_chapters_mp4(audio_path, sorted_chapters)
    else:
        raise ChapterWriteError(
            f"Cannot write chapters to {audio_path}: unsupported audio "
            f"container '{container}'"
        )


class ChapterWriteError(Exception):
    """Raised when chapter metadata cannot be embedded into an audio file."""


def _write_chapters_mp3(audio_path: str, sorted_chapters: list[Chapter]) -> None:
    """Embed chapters as ID3 CHAP/CTOC frames in an MP3 file."""
    try:
        audio = MP3(audio_path)

        # Create ID3 tags if they don't exist
        if audio.tags is None:
            audio.add_tags()
        tags = cast(ID3, audio.tags)

        # Remove existing chapter frames
        keys_to_remove = [
            key for key in tags.keys() if key.startswith(("CHAP", "CTOC"))
        ]
        for key in keys_to_remove:
            del tags[key]

        # Add new chapter frames
        chapter_ids = []
        for i, chapter in enumerate(sorted_chapters):
            element_id = f"chp{i}"
            chapter_ids.append(element_id)

            # Create TIT2 sub-frame for chapter title
            tit2 = TIT2(encoding=3, text=[chapter.title])

            # Create CHAP frame
            chap = CHAP(
                element_id=element_id,
                start_time=chapter.start_time_ms,
                end_time=chapter.end_time_ms,
                start_offset=0xFFFFFFFF,  # Not used
                end_offset=0xFFFFFFFF,  # Not used
                sub_frames=[tit2],
            )
            tags.add(chap)

        # Create CTOC (Table of Contents) frame
        if chapter_ids:
            ctoc = CTOC(
                element_id="toc",
                flags=3,  # Top-level, ordered
                child_element_ids=chapter_ids,
                sub_frames=[],
            )
            tags.add(ctoc)

        audio.save()

        logger.info("Wrote %d chapters to %s", len(sorted_chapters), audio_path)

    except Exception as e:
        logger.error("Failed to write chapters to %s: %s", audio_path, e)
        raise ChapterWriteError(str(e)) from e


def _build_ffmetadata(sorted_chapters: list[Chapter]) -> str:
    """Render chapters as an ffmpeg FFMETADATA document (millisecond timebase)."""
    lines = [";FFMETADATA1"]
    for chapter in sorted_chapters:
        # Escape the characters ffmpeg treats as metadata syntax.
        title = (
            chapter.title.replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
            .replace("\n", "\\\n")
        )
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={max(0, chapter.start_time_ms)}",
            f"END={max(0, chapter.end_time_ms)}",
            f"title={title}",
        ]
    return "\n".join(lines) + "\n"


def _write_chapters_mp4(audio_path: str, sorted_chapters: list[Chapter]) -> None:
    """Embed chapters into an MP4/M4A file via an ffmpeg FFMETADATA remux.

    ffmpeg cannot edit in place, so we remux (``-c copy``, lossless) into a
    sibling temp file and atomically replace the original. ``-map_metadata 0``
    preserves the existing tags/artwork while ``-map_chapters 1`` injects the
    new chapters.
    """
    directory = os.path.dirname(audio_path) or "."
    meta_fd, meta_path = tempfile.mkstemp(suffix=".ffmeta", dir=directory)
    tmp_output = f"{audio_path}.chapters.tmp.m4a"
    try:
        with os.fdopen(meta_fd, "w", encoding="utf-8") as meta_file:
            meta_file.write(_build_ffmetadata(sorted_chapters))

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", audio_path,
                    "-i", meta_path,
                    "-map", "0",
                    "-map_metadata", "0",
                    "-map_chapters", "1",
                    "-c", "copy",
                    tmp_output,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error(
                "ffmpeg failed to write chapters to %s: %s",
                audio_path,
                exc.stderr.strip() if exc.stderr else exc,
            )
            raise ChapterWriteError(
                f"ffmpeg failed to embed chapters in {audio_path}"
            ) from exc

        os.replace(tmp_output, audio_path)
        logger.info("Wrote %d chapters to %s", len(sorted_chapters), audio_path)
    finally:
        for leftover in (meta_path, tmp_output):
            try:
                os.remove(leftover)
            except OSError:
                pass


def write_adjusted_chapters(
    audio_path: str,
    chapters_to_keep: list[Chapter],
    removed_segments: list[tuple[float, float]],
) -> None:
    """
    Write chapters to an MP3 file with timestamps adjusted for removed segments.

    Convenience function that combines recalculation and writing.

    Args:
        audio_path: Path to the MP3 file
        chapters_to_keep: List of chapters that were not removed as ads
        removed_segments: List of (start_sec, end_sec) tuples that were removed
    """
    adjusted_chapters = recalculate_chapter_times(chapters_to_keep, removed_segments)
    write_chapters(audio_path, adjusted_chapters)
