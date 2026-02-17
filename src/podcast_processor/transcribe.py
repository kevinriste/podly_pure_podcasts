import json
import logging
import shutil
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from groq import Groq
from openai import OpenAI
from openai.types.audio.transcription_segment import TranscriptionSegment
from pydantic import BaseModel

from podcast_processor.audio import split_audio
from shared.config import GroqWhisperConfig, RemoteWhisperConfig


class Segment(BaseModel):
    start: float
    end: float
    text: str


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

    def transcribe(self, _: str) -> list[Segment]:
        self.logger.info("Using test whisper")
        segments = [
            Segment(start=0, end=1, text="This is a test"),
            Segment(start=1, end=2, text="This is another test"),
        ]
        self.last_raw_response_body = json.dumps(
            [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in segments
            ],
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
            import whisper  # type: ignore[import-untyped]
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
        self.last_raw_response_body = _combine_chunk_raw_response_bodies(
            raw_chunk_bodies
        )
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

    def get_segments_for_chunk(
        self, chunk_path: str
    ) -> tuple[list[TranscriptionSegment], str]:
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

        chunks = split_audio(
            Path(audio_file_path), Path(audio_chunk_path), 12 * 1024 * 1024
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
        self.last_raw_response_body = _combine_chunk_raw_response_bodies(
            raw_chunk_bodies
        )
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

    def get_segments_for_chunk(
        self, chunk_path: str
    ) -> tuple[list[GroqTranscriptionSegment], str]:

        self.logger.info("[GROQ_API_CALL] Sending chunk to Groq API: %s", chunk_path)
        transcription = self.client.audio.transcriptions.create(
            file=Path(chunk_path),
            model=self.config.model,
            response_format="verbose_json",  # Ensure segments are included
            language=self.config.language,
        )
        self.logger.info(
            "[GROQ_API_CALL] Received response from Groq API for: %s", chunk_path
        )
        raw_body = _serialize_raw_response_body(transcription)

        if transcription.segments is None:  # type: ignore [attr-defined]
            self.logger.warning(
                "[GROQ_API_CALL] No segments found in transcription for %s", chunk_path
            )
            return [], raw_body

        groq_segments = [
            GroqTranscriptionSegment(
                start=seg["start"], end=seg["end"], text=seg["text"]
            )
            for seg in transcription.segments  # type: ignore [attr-defined]
        ]

        self.logger.info(
            "[GROQ_API_CALL] Got %d segments from chunk", len(groq_segments)
        )
        return groq_segments, raw_body
