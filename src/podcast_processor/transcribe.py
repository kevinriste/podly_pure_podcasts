import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

from groq import Groq
from openai import OpenAI
from openai.types.audio.transcription_segment import TranscriptionSegment
from pydantic import BaseModel

import requests

from podcast_processor.audio import split_audio
from shared.config import GroqWhisperConfig, RemoteWhisperConfig, WhisperXConfig


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class Transcriber(ABC):
    last_raw_response_body: str | None = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> list[Segment]:
        pass


def _serialize_raw_response_body(payload: Any) -> str:
    """Serialize raw provider payload to a JSON-ish string for stats/debugging."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            return payload.decode("utf-8", errors="replace")

    model_dump_json = getattr(payload, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            return str(model_dump_json(indent=2))
        except TypeError:
            return str(model_dump_json())

    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(), ensure_ascii=True, default=str, indent=2)

    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        return json.dumps(to_dict(), ensure_ascii=True, default=str, indent=2)

    return json.dumps(payload, ensure_ascii=True, default=str, indent=2)


def _combine_chunk_raw_response_bodies(raw_bodies: list[str]) -> str:
    if not raw_bodies:
        return ""
    if len(raw_bodies) == 1:
        return raw_bodies[0]

    lines: list[str] = []
    for idx, raw_body in enumerate(raw_bodies, start=1):
        lines.append(f"--- chunk {idx} ---")
        lines.append(raw_body)
    return "\n".join(lines)


class LocalTranscriptSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    def to_segment(self) -> Segment:
        return Segment(start=self.start, end=self.end, text=self.text)


class TestWhisperTranscriber(Transcriber):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @property
    def model_name(self) -> str:
        return "test_whisper"

    def transcribe(self, audio_file_path: str) -> list[Segment]:
        del audio_file_path
        self.logger.info("Using test whisper")
        segments = [
            Segment(start=0, end=1, text="This is a test"),
            Segment(start=1, end=2, text="This is another test"),
        ]
        self.last_raw_response_body = json.dumps(
            [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments],
            ensure_ascii=True,
            indent=2,
        )
        return segments


class LocalWhisperTranscriber(Transcriber):
    def __init__(self, logger: logging.Logger, whisper_model: str):
        self.logger = logger
        self.whisper_model = whisper_model

    @property
    def model_name(self) -> str:
        return f"local_{self.whisper_model}"

    @staticmethod
    def convert_to_pydantic(
        transcript_data: list[Any],
    ) -> list[LocalTranscriptSegment]:
        return [LocalTranscriptSegment(**item) for item in transcript_data]

    @staticmethod
    def local_seg_to_seg(local_segments: list[LocalTranscriptSegment]) -> list[Segment]:
        return [seg.to_segment() for seg in local_segments]

    def transcribe(self, audio_file_path: str) -> list[Segment]:
        # Import whisper only when needed to avoid CUDA dependencies during module import
        try:
            import whisper
        except ImportError as e:
            self.logger.error(f"Failed to import whisper: {e}")
            raise ImportError(
                "whisper library is required for LocalWhisperTranscriber"
            ) from e

        self.logger.info("Using local whisper")
        models = whisper.available_models()
        self.logger.info(f"Available models: {models}")

        model = whisper.load_model(name=self.whisper_model)

        self.logger.info("Beginning transcription")
        start = time.time()
        result = model.transcribe(audio_file_path, fp16=False, language="English")
        end = time.time()
        elapsed = end - start
        self.logger.info(f"Transcription completed in {elapsed}")
        self.last_raw_response_body = _serialize_raw_response_body(result)
        segments = result["segments"]
        typed_segments = self.convert_to_pydantic(segments)

        return self.local_seg_to_seg(typed_segments)


class OpenAIWhisperTranscriber(Transcriber):
    def __init__(self, logger: logging.Logger, config: RemoteWhisperConfig):
        self.logger = logger
        self.config = config

        self.openai_client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_sec,
        )

    @property
    def model_name(self) -> str:
        return self.config.model  # e.g. "whisper-1"

    def transcribe(self, audio_file_path: str) -> list[Segment]:
        self.logger.info(
            "[WHISPER_REMOTE] Starting remote whisper transcription for: %s",
            audio_file_path,
        )
        audio_chunk_path = audio_file_path + "_parts"

        chunks = split_audio(
            Path(audio_file_path),
            Path(audio_chunk_path),
            self.config.chunksize_mb * 1024 * 1024,
        )

        self.logger.info("[WHISPER_REMOTE] Processing %d chunks", len(chunks))
        all_segments: list[TranscriptionSegment] = []
        raw_chunk_bodies: list[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_path, offset = chunk
            self.logger.info(
                "[WHISPER_REMOTE] Processing chunk %d/%d: %s",
                idx + 1,
                len(chunks),
                chunk_path,
            )
            segments, raw_body = self.get_segments_for_chunk(str(chunk_path))
            raw_chunk_bodies.append(raw_body)
            self.logger.info(
                "[WHISPER_REMOTE] Chunk %d/%d complete: %d segments",
                idx + 1,
                len(chunks),
                len(segments),
            )
            all_segments.extend(self.add_offset_to_segments(segments, offset))

        shutil.rmtree(audio_chunk_path)
        self.logger.info(
            "[WHISPER_REMOTE] Transcription complete: %d total segments",
            len(all_segments),
        )
        self.last_raw_response_body = _combine_chunk_raw_response_bodies(raw_chunk_bodies)
        return self.convert_segments(all_segments)

    @staticmethod
    def convert_segments(segments: list[TranscriptionSegment]) -> list[Segment]:
        return [
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for seg in segments
        ]

    @staticmethod
    def add_offset_to_segments(
        segments: list[TranscriptionSegment], offset_ms: int
    ) -> list[TranscriptionSegment]:
        offset_sec = float(offset_ms) / 1000.0
        for segment in segments:
            segment.start += offset_sec
            segment.end += offset_sec

        return segments

    def get_segments_for_chunk(self, chunk_path: str) -> tuple[list[TranscriptionSegment], str]:
        with open(chunk_path, "rb") as f:
            self.logger.info(
                "[WHISPER_API_CALL] Sending chunk to API: %s (timeout=%ds)",
                chunk_path,
                self.config.timeout_sec,
            )

            transcription = self.openai_client.audio.transcriptions.create(
                model=self.config.model,
                file=f,
                timestamp_granularities=["segment"],
                language=self.config.language,
                response_format="verbose_json",
            )

            self.logger.debug("Got transcription")
            raw_body = _serialize_raw_response_body(transcription)

            segments = transcription.segments
            assert segments is not None

            self.logger.debug(f"Got {len(segments)} segments")

            return segments, raw_body


class GroqTranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class GroqWhisperTranscriber(Transcriber):
    def __init__(self, logger: logging.Logger, config: GroqWhisperConfig):
        self.logger = logger
        self.config = config
        self.client = Groq(
            api_key=config.api_key,
            max_retries=config.max_retries,
        )

    @property
    def model_name(self) -> str:
        return f"groq_{self.config.model}"

    def transcribe(self, audio_file_path: str) -> list[Segment]:
        self.logger.info(
            "[WHISPER_GROQ] Starting Groq whisper transcription for: %s",
            audio_file_path,
        )
        audio_chunk_path = audio_file_path + "_parts"

        # 12MB seems to cause instability in Groq
        chunks = split_audio(
            Path(audio_file_path), Path(audio_chunk_path), 6 * 1024 * 1024
        )

        self.logger.info("[WHISPER_GROQ] Processing %d chunks", len(chunks))
        all_segments: list[GroqTranscriptionSegment] = []
        raw_chunk_bodies: list[str] = []

        for idx, chunk in enumerate(chunks):
            chunk_path, offset = chunk
            self.logger.info(
                "[WHISPER_GROQ] Processing chunk %d/%d: %s",
                idx + 1,
                len(chunks),
                chunk_path,
            )
            segments, raw_body = self.get_segments_for_chunk(str(chunk_path))
            raw_chunk_bodies.append(raw_body)
            self.logger.info(
                "[WHISPER_GROQ] Chunk %d/%d complete: %d segments",
                idx + 1,
                len(chunks),
                len(segments),
            )
            all_segments.extend(self.add_offset_to_segments(segments, offset))

        shutil.rmtree(audio_chunk_path)
        self.logger.info(
            "[WHISPER_GROQ] Transcription complete: %d total segments",
            len(all_segments),
        )
        self.last_raw_response_body = _combine_chunk_raw_response_bodies(raw_chunk_bodies)
        return self.convert_segments(all_segments)

    @staticmethod
    def convert_segments(segments: list[GroqTranscriptionSegment]) -> list[Segment]:
        return [
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
            for seg in segments
        ]

    @staticmethod
    def add_offset_to_segments(
        segments: list[GroqTranscriptionSegment], offset_ms: int
    ) -> list[GroqTranscriptionSegment]:
        offset_sec = float(offset_ms) / 1000.0
        for segment in segments:
            segment.start += offset_sec
            segment.end += offset_sec

        return segments

    def get_segments_for_chunk(self, chunk_path: str) -> tuple[list[GroqTranscriptionSegment], str]:
        retries = self.config.max_retries if self.config.max_retries is not None else 0
        max_attempts = retries + 1
        for attempt in range(1, max_attempts + 1):
            self.logger.info(
                "[GROQ_API_CALL] Sending chunk to Groq API: %s (attempt %d/%d)",
                chunk_path,
                attempt,
                max_attempts,
            )
            try:
                transcription = self.client.audio.transcriptions.create(
                    file=Path(chunk_path),
                    model=self.config.model,
                    response_format="verbose_json",  # Ensure segments are included
                    language=self.config.language,
                )
            except Exception as exc:
                self.logger.warning(
                    "[GROQ_API_CALL] Attempt %d/%d failed for %s: %s",
                    attempt,
                    max_attempts,
                    chunk_path,
                    exc,
                )
                if attempt == max_attempts:
                    raise
                time.sleep(1.5**attempt)
                continue

            self.logger.info(
                "[GROQ_API_CALL] Received response from Groq API for: %s (attempt %d/%d)",
                chunk_path,
                attempt,
                max_attempts,
            )
            raw_body = _serialize_raw_response_body(transcription)

            if transcription.segments is None:  # type: ignore [attr-defined]
                self.logger.warning(
                    "[GROQ_API_CALL] No segments found in transcription for %s",
                    chunk_path,
                )
                return [], raw_body

            groq_segments = [
                GroqTranscriptionSegment(
                    start=seg["start"], end=seg["end"], text=seg["text"]
                )
                for seg in transcription.segments  # type: ignore [attr-defined]
            ]

            self.logger.info(
                "[GROQ_API_CALL] Got %d segments from chunk (attempt %d/%d)",
                len(groq_segments),
                attempt,
                max_attempts,
            )
            return groq_segments, raw_body

        # unreachable, but satisfies type checker
        return [], ""


class WhisperXTranscriber(Transcriber):
    """Transcriber using the WhisperX API with diarization and alignment."""

    def __init__(self, logger: logging.Logger, config: WhisperXConfig):
        self.logger = logger
        self.config = config

    @property
    def model_name(self) -> str:
        return f"whisperx_{self.config.model}"

    @staticmethod
    def _dominant_speaker(words: list[dict[str, Any]]) -> str | None:
        """Extract the most common speaker label from word-level data."""
        speakers = [w["speaker"] for w in words if w.get("speaker")]
        if not speakers:
            return None
        counter = Counter(speakers)
        return counter.most_common(1)[0][0]

    def transcribe(self, audio_file_path: str) -> list[Segment]:
        self.logger.info(
            "[WHISPERX] Starting WhisperX transcription for: %s",
            audio_file_path,
        )

        url = f"{self.config.base_url}/audio/transcriptions"

        with open(audio_file_path, "rb") as f:
            self.logger.info(
                "[WHISPERX_API_CALL] Sending file to API: %s (timeout=%ds)",
                audio_file_path,
                self.config.timeout_sec,
            )
            response = requests.post(
                url,
                files={"file": f},
                data={
                    "model": self.config.model,
                    "language": self.config.language,
                    "response_format": "verbose_json",
                    "diarize": "true",
                    "align": "true",
                    "timestamp_granularities[]": ["word", "segment"],
                },
                timeout=self.config.timeout_sec,
            )

        response.raise_for_status()
        data = response.json()
        self.last_raw_response_body = _serialize_raw_response_body(data)

        # WhisperX nests segments under response['segments']['segments']
        segments_container = data.get("segments", {})
        if isinstance(segments_container, dict):
            raw_segments = segments_container.get("segments", [])
        elif isinstance(segments_container, list):
            # Some versions may return a flat list
            raw_segments = segments_container
        else:
            raw_segments = []

        self.logger.info(
            "[WHISPERX] Received %d segments from API",
            len(raw_segments),
        )

        segments: list[Segment] = []
        for raw_seg in raw_segments:
            start = float(raw_seg.get("start", 0.0))
            end = float(raw_seg.get("end", 0.0))
            text = str(raw_seg.get("text", ""))

            # Extract dominant speaker from word-level data
            words = raw_seg.get("words", [])
            speaker = self._dominant_speaker(words) if words else None

            segments.append(Segment(
                start=start,
                end=end,
                text=text,
                speaker=speaker,
            ))

        self.logger.info(
            "[WHISPERX] Transcription complete: %d total segments",
            len(segments),
        )
        return segments
