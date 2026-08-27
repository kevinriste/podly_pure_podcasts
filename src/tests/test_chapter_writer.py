import shutil
import subprocess

import pytest

from podcast_processor.chapter_reader import Chapter, read_chapters
from podcast_processor.chapter_writer import (
    ChapterWriteError,
    recalculate_chapter_times,
    write_chapters,
)

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _generate_audio(path: str, codec: str) -> str:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:a", codec, path,
        ],
        check=True,
    )
    return path


_SAMPLE_CHAPTERS = [
    Chapter("c0", "Intro", 0, 1500),
    Chapter("c1", "Main", 1500, 3000),
]


@ffmpeg_required
def test_write_chapters_roundtrip_mp3(tmp_path) -> None:
    path = _generate_audio(str(tmp_path / "ep.mp3"), "libmp3lame")
    write_chapters(path, _SAMPLE_CHAPTERS)

    titles = [c.title for c in read_chapters(path)]
    assert titles == ["Intro", "Main"]


@ffmpeg_required
def test_write_chapters_roundtrip_m4a(tmp_path) -> None:
    path = _generate_audio(str(tmp_path / "ep.m4a"), "aac")
    write_chapters(path, _SAMPLE_CHAPTERS)

    titles = [c.title for c in read_chapters(path)]
    assert titles == ["Intro", "Main"]


@ffmpeg_required
def test_write_chapters_handles_m4a_mislabeled_as_mp3(tmp_path) -> None:
    """The incident case: Patreon M4A served under a .mp3 name must not crash."""
    real_m4a = _generate_audio(str(tmp_path / "ep.m4a"), "aac")
    path = str(tmp_path / "ep.mp3")
    shutil.move(real_m4a, path)  # M4A bytes, .mp3 name
    write_chapters(path, _SAMPLE_CHAPTERS)

    titles = [c.title for c in read_chapters(path)]
    assert titles == ["Intro", "Main"]


def test_write_chapters_raises_on_unsupported_container(tmp_path) -> None:
    path = str(tmp_path / "ep.mp3")
    with open(path, "wb") as handle:
        handle.write(b"not real audio")
    with pytest.raises(ChapterWriteError):
        write_chapters(path, _SAMPLE_CHAPTERS)


def test_recalculate_chapter_times_shrinks_chapter_when_cut_occurs_inside() -> None:
    chapters = [
        Chapter("c1", "Long section", 0, 600_000),
        Chapter("c2", "Later section", 600_000, 900_000),
    ]

    adjusted = recalculate_chapter_times(chapters, removed_segments=[(100.0, 130.0)])

    assert [c.start_time_ms for c in adjusted] == [0, 570_000]
    assert [c.end_time_ms for c in adjusted] == [570_000, 870_000]


def test_recalculate_chapter_times_offsets_each_marker_by_prior_removed_audio() -> None:
    chapters = [
        Chapter("c1", "Intro", 0, 120_000),
        Chapter("c2", "Segment A", 120_000, 240_000),
        Chapter("c3", "Segment B", 240_000, 360_000),
    ]

    adjusted = recalculate_chapter_times(
        chapters,
        removed_segments=[
            (30.0, 40.0),
            (150.0, 170.0),
        ],
    )

    assert [c.start_time_ms for c in adjusted] == [0, 110_000, 210_000]
    assert [c.end_time_ms for c in adjusted] == [110_000, 210_000, 330_000]


def test_recalculate_chapter_times_merges_overlapping_removed_windows() -> None:
    chapters = [
        Chapter("c1", "Part 1", 0, 400_000),
        Chapter("c2", "Part 2", 400_000, 800_000),
    ]

    adjusted = recalculate_chapter_times(
        chapters,
        removed_segments=[
            (100.0, 180.0),
            (150.0, 220.0),
        ],
    )

    # Unique removed duration is 120 seconds, not 150 seconds.
    assert [c.start_time_ms for c in adjusted] == [0, 280_000]
    assert [c.end_time_ms for c in adjusted] == [280_000, 680_000]
