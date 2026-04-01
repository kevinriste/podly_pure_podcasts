"""Client for the inaSpeechSegmenter API."""

import json
import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger("global_logger")


@dataclass
class AudioSegmentResult:
    label: str
    start_time: float
    end_time: float


def analyze_audio(
    audio_path: str, base_url: str, timeout: int = 3600
) -> tuple[list[AudioSegmentResult], str]:
    """Send audio to INA API and return segmentation results.

    Returns:
        A tuple of (results, raw_json_response).
    """
    url = f"{base_url}/segment"

    with open(audio_path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            timeout=timeout,
        )

    response.raise_for_status()
    data = response.json()

    raw_response = json.dumps(data, ensure_ascii=True, default=str, indent=2)

    results = [
        AudioSegmentResult(
            label=seg["label"],
            start_time=seg["start"],
            end_time=seg["end"],
        )
        for seg in data
    ]

    return results, raw_response
