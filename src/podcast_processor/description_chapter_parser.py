"""Parse chapter timestamps from episode descriptions."""

from __future__ import annotations

import html
import logging
import re

from podcast_processor.chapter_reader import Chapter

logger = logging.getLogger("global_logger")

_TIMESTAMP_RE = re.compile(r"(?<!\d)(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})(?!\d)")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_CLOSE_RE = re.compile(r"</(?:p|div|li|ul|ol|h[1-6]|tr|td|blockquote)>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(((?:https?://)|(?:www\.))[^)]+\)")
_URL_RE = re.compile(r"(?:(?:https?://)|(?:www\.))\S+", re.IGNORECASE)


def parse_chapters_from_description(
    description: str | None,
    *,
    total_duration_ms: int | None = None,
    min_chapters: int = 2,
) -> list[Chapter]:
    """
    Parse chapter markers from episode description text.

    Returns an empty list if no high-confidence chapter list is detected.
    """
    if not description:
        return []

    lines = _description_lines(description)
    candidates: list[tuple[int, int, str]] = []

    for line_no, line in enumerate(lines):
        match = _TIMESTAMP_RE.search(line)
        if not match:
            continue

        ts_raw = match.group("ts")
        start_ms = _parse_timestamp_ms(ts_raw)
        if start_ms is None:
            continue

        title = _extract_title(line, match)
        if not title:
            title = f"Chapter {len(candidates) + 1}"

        candidates.append((line_no, start_ms, title))

    if not candidates:
        return []

    unique_candidates = _select_monotonic_candidates(candidates)
    if len(unique_candidates) < min_chapters:
        return []

    if total_duration_ms is not None and total_duration_ms <= unique_candidates[-1][1]:
        logger.debug(
            (
                "Description chapter parse rejected: "
                "total_duration_ms=%s <= last chapter %s"
            ),
            total_duration_ms,
            unique_candidates[-1][1],
        )
        return []

    chapters: list[Chapter] = []
    for index, (_line_no, start_ms, title) in enumerate(unique_candidates):
        if index + 1 < len(unique_candidates):
            end_ms = unique_candidates[index + 1][1]
        else:
            end_ms = total_duration_ms if total_duration_ms is not None else start_ms

        end_ms = max(start_ms, end_ms)
        chapters.append(
            Chapter(
                element_id=f"desc{index}",
                title=title,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            )
        )

    return chapters


def _description_lines(description: str) -> list[str]:
    text = description
    text = _BR_RE.sub("\n", text)
    text = _BLOCK_CLOSE_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)

    normalized_lines: list[str] = []
    for raw in text.splitlines():
        line = _SPACE_RE.sub(" ", raw).strip()
        if line:
            normalized_lines.append(line)
    return normalized_lines


def _parse_timestamp_ms(value: str) -> int | None:
    parts = value.split(":")
    if len(parts) == 2:
        hours = 0
        minutes_str, seconds_str = parts
    elif len(parts) == 3:
        hours_str, minutes_str, seconds_str = parts
        try:
            hours = int(hours_str)
        except ValueError:
            return None
    else:
        return None

    try:
        minutes = int(minutes_str)
        seconds = int(seconds_str)
    except ValueError:
        return None

    if minutes < 0 or seconds < 0 or seconds >= 60:
        return None
    if len(parts) == 3 and minutes >= 60:
        return None

    return ((hours * 3600) + (minutes * 60) + seconds) * 1000


def _extract_title(line: str, match: re.Match[str]) -> str:
    start, end = match.span("ts")
    before = line[:start]
    after = line[end:]

    # Prefer the text after the timestamp, but support "Title - 00:00" formats.
    trailing = _clean_title_fragment(after)
    if trailing:
        return trailing

    leading = _clean_title_fragment(before)
    return leading


def _clean_title_fragment(fragment: str) -> str:
    cleaned = fragment.strip()

    # Remove common wrappers / separators around timestamps.
    cleaned = re.sub(r"^[\s\-\u2013\u2014:|>\]\)\.]+", "", cleaned)
    cleaned = re.sub(r"[\[\(\-:\u2013\u2014|<\s]+$", "", cleaned)

    # Remove common list bullets / numbering prefixes from remaining text.
    cleaned = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", cleaned)

    # Preserve markdown link text while dropping the URL target.
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)

    # Description-based chapter titles should be concise titles. Raw URLs in titles
    # are often extracted from show notes and can break chapter UIs in players.
    cleaned = _URL_RE.sub("", cleaned)

    # Trim separators left behind after URL removal.
    cleaned = re.sub(r"[\s\-\u2013\u2014:|>]+$", "", cleaned)

    cleaned = _SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _select_monotonic_candidates(
    candidates: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    if not candidates:
        return []

    # Pick the best strictly increasing sequence of timestamps while preserving line
    # order. This tolerates noisy extra timestamp lines elsewhere in the description.
    count = len(candidates)
    lengths = [1] * count
    prev_idx = [-1] * count

    for i in range(count):
        _line_i, start_i, _title_i = candidates[i]
        for j in range(i):
            _line_j, start_j, _title_j = candidates[j]
            if start_j >= start_i:
                continue
            next_len = lengths[j] + 1
            if next_len > lengths[i]:
                lengths[i] = next_len
                prev_idx[i] = j
                continue
            if next_len < lengths[i]:
                continue

            # Tie-break: prefer the more compact sequence by line span.
            current_chain = _reconstruct_candidate_sequence(candidates, prev_idx, i)
            challenger_chain = _reconstruct_candidate_sequence_with_tail(
                candidates,
                prev_idx,
                j,
                candidates[i],
            )
            if _candidate_line_span(challenger_chain) < _candidate_line_span(
                current_chain
            ):
                prev_idx[i] = j

    best_idx = max(range(count), key=lambda idx: (lengths[idx], -candidates[idx][0]))
    best = _reconstruct_candidate_sequence(candidates, prev_idx, best_idx)

    if len(best) < len(candidates):
        logger.debug(
            "Description chapter parse ignored %d non-monotonic/duplicate timestamp "
            "candidate lines and kept %d monotonic chapters",
            len(candidates) - len(best),
            len(best),
        )

    # Remove duplicate start times within the selected chain (should be rare with
    # strict increase, but keep this for defensive normalization).
    deduped: list[tuple[int, int, str]] = []
    seen_starts: set[int] = set()
    for line_no, start_ms, title in best:
        if start_ms in seen_starts:
            continue
        deduped.append((line_no, start_ms, title))
        seen_starts.add(start_ms)
    return deduped


def _reconstruct_candidate_sequence(
    candidates: list[tuple[int, int, str]],
    prev_idx: list[int],
    end_idx: int,
) -> list[tuple[int, int, str]]:
    chain: list[tuple[int, int, str]] = []
    cursor = end_idx
    while cursor != -1:
        chain.append(candidates[cursor])
        cursor = prev_idx[cursor]
    chain.reverse()
    return chain


def _reconstruct_candidate_sequence_with_tail(
    candidates: list[tuple[int, int, str]],
    prev_idx: list[int],
    end_idx: int,
    tail: tuple[int, int, str],
) -> list[tuple[int, int, str]]:
    chain = _reconstruct_candidate_sequence(candidates, prev_idx, end_idx)
    chain.append(tail)
    return chain


def _candidate_line_span(candidates: list[tuple[int, int, str]]) -> int:
    if not candidates:
        return 0
    return int(candidates[-1][0]) - int(candidates[0][0])
