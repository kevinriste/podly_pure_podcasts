"""Build merged transcript + audio stream for enriched ad detection."""

import logging
from typing import Any

logger = logging.getLogger("global_logger")


def build_merged_stream(
    transcript_segments: list[Any],
    audio_segments: list[Any],
    min_music_duration: float = 0.5,
    min_silence_duration: float = 2.0,
) -> str:
    """Build a merged stream of speech + music/silence markers, sorted by time.

    Only includes music and silence (noEnergy) markers above minimum durations.
    Speech segments from INA are excluded since we have the full transcript.
    """
    items: list[tuple[float, float, str]] = []  # (start, end, formatted_line)

    # Add transcript segments with speaker labels
    for seg in transcript_segments:
        speaker = getattr(seg, "speaker", None) or "?"
        text = seg.text.strip() if seg.text else ""
        items.append((
            seg.start_time,
            seg.end_time,
            f"[{seg.start_time:.1f}-{seg.end_time:.1f}] {speaker}: {text}",
        ))

    # Add music and silence markers from INA
    for aseg in audio_segments:
        dur = aseg.end_time - aseg.start_time
        if aseg.label == "music" and dur >= min_music_duration:
            items.append((
                aseg.start_time,
                aseg.end_time,
                f"[{aseg.start_time:.1f}-{aseg.end_time:.1f}] [MUSIC] ({dur:.1f}s)",
            ))
        elif aseg.label == "noEnergy" and dur >= min_silence_duration:
            items.append((
                aseg.start_time,
                aseg.end_time,
                f"[{aseg.start_time:.1f}-{aseg.end_time:.1f}] [SILENCE] ({dur:.1f}s)",
            ))

    # Sort by start time
    items.sort(key=lambda x: x[0])

    return "\n".join(line for _, _, line in items)
