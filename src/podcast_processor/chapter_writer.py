"""Write chapter metadata to processed MP3 files with adjusted timestamps."""

import logging
from typing import cast

from mutagen.id3 import CHAP, CTOC, ID3, TIT2
from mutagen.mp3 import MP3

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

    # Sort and normalize removed segments to integer milliseconds
    sorted_segments_ms = sorted(
        [(int(start * 1000), int(end * 1000)) for start, end in removed_segments],
        key=lambda x: x[0],
    )

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
    Write chapter metadata to an MP3 file.

    Overwrites any existing chapter data in the file.

    Args:
        audio_path: Path to the MP3 file
        chapters: List of Chapter objects to write
    """
    if not chapters:
        logger.info("No chapters to write to %s", audio_path)
        return

    # Sort chapters by start time to ensure correct order
    sorted_chapters = sorted(chapters, key=lambda c: c.start_time_ms)

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

        logger.info("Wrote %d chapters to %s", len(chapters), audio_path)

    except Exception as e:
        logger.error("Failed to write chapters to %s: %s", audio_path, e)
        raise


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
