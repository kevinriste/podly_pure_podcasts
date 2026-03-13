from podcast_processor.chapter_reader import Chapter
from podcast_processor.chapter_writer import recalculate_chapter_times


def test_recalculate_chapter_times_shrinks_chapter_when_cut_occurs_inside() -> None:
    chapters = [
        Chapter("c1", "Long section", 0, 600_000),
        Chapter("c2", "Later section", 600_000, 900_000),
    ]

    adjusted = recalculate_chapter_times(chapters, removed_segments=[(100.0, 130.0)])

    assert [c.start_time_ms for c in adjusted] == [0, 570_000]
    assert [c.end_time_ms for c in adjusted] == [570_000, 870_000]
