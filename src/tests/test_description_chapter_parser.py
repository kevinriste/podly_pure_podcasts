from podcast_processor.description_chapter_parser import parse_chapters_from_description


def test_parse_description_chapters_markdown_list_infers_end_times() -> None:
    description = """
    00:00 Intro
    03:12 News and follow-up
    12:34 Q&A
    """

    chapters = parse_chapters_from_description(description, total_duration_ms=1_000_000)

    assert [ch.title for ch in chapters] == ["Intro", "News and follow-up", "Q&A"]
    assert [ch.start_time_ms for ch in chapters] == [0, 192_000, 754_000]
    assert [ch.end_time_ms for ch in chapters] == [192_000, 754_000, 1_000_000]


def test_parse_description_chapters_html_formats() -> None:
    description = (
        "<p>[00:00] Cold open</p><p>05:10 - Main topic</p><p>1:02:03 Wrap-up</p>"
    )

    chapters = parse_chapters_from_description(description, total_duration_ms=4_000_000)

    assert [ch.title for ch in chapters] == ["Cold open", "Main topic", "Wrap-up"]
    assert [ch.start_time_ms for ch in chapters] == [0, 310_000, 3_723_000]
    assert chapters[-1].end_time_ms == 4_000_000


def test_parse_description_chapters_supports_title_before_timestamp() -> None:
    description = """
    Intro - 00:00
    Sponsor mention (05:00)
    Deep dive | 10:30
    """

    chapters = parse_chapters_from_description(description)

    assert [ch.title for ch in chapters] == ["Intro", "Sponsor mention", "Deep dive"]
    assert [ch.start_time_ms for ch in chapters] == [0, 300_000, 630_000]
    assert chapters[-1].end_time_ms == 630_000


def test_parse_description_chapters_rejects_single_timestamp() -> None:
    description = "This episode runs 52:14 and covers a lot."

    assert parse_chapters_from_description(description) == []


def test_parse_description_chapters_rejects_non_monotonic_timestamps() -> None:
    description = """
    10:00 Segment A
    05:00 Segment B
    """

    assert parse_chapters_from_description(description) == []


def test_parse_description_chapters_ignores_non_monotonic_timestamp_noise() -> None:
    description = """
    00:00 Intro
    05:00 Main topic
    10:00 Wrap-up

    Links:
    Patreon shoutout at 01:30
    Bonus clip starts 00:45
    """

    chapters = parse_chapters_from_description(description, total_duration_ms=900_000)

    assert [ch.title for ch in chapters] == ["Intro", "Main topic", "Wrap-up"]
    assert [ch.start_time_ms for ch in chapters] == [0, 300_000, 600_000]
    assert [ch.end_time_ms for ch in chapters] == [300_000, 600_000, 900_000]


def test_parse_description_chapters_prefers_larger_monotonic_sequence() -> None:
    description = """
    Promo mention 12:00
    Chapters:
    00:00 Intro
    03:00 Story one
    08:00 Story two
    15:00 Wrap-up
    """

    chapters = parse_chapters_from_description(description, total_duration_ms=1_200_000)

    assert [ch.title for ch in chapters] == [
        "Intro",
        "Story one",
        "Story two",
        "Wrap-up",
    ]
    assert [ch.start_time_ms for ch in chapters] == [0, 180_000, 480_000, 900_000]


def test_parse_description_chapters_strips_urls_from_titles() -> None:
    description = """
    00:00 Intro
    01:35 My husband keeps sending flirty emojis to my cousin https://www.reddit.com/r/TwoHotTakes/comments/1obdbxi/foo/
    08:46 Sponsor
    09:42 [Story title with link](https://example.com/post)
    """

    chapters = parse_chapters_from_description(description, total_duration_ms=900_000)

    assert [ch.title for ch in chapters] == [
        "Intro",
        "My husband keeps sending flirty emojis to my cousin",
        "Sponsor",
        "Story title with link",
    ]
