# INA Speech Segmenter & Speaker Diarization

This document describes the INA speech segmenter integration and WhisperX diarization
feature added in this branch.

## Overview

Two complementary analysis capabilities run in parallel with the existing transcription
pipeline:

1. **INA speech segmenter** — classifies every second of audio as `speech`, `music`, or
   `noise/silence` by calling an external INA API. Results are stored in the new
   `audio_segment` table and visualised in the Stats UI.

2. **WhisperX diarization** — replaces the plain Whisper API call with a WhisperX request
   (`diarize=true`, `align=true`). WhisperX returns word-level speaker labels; the
   transcriber extracts the dominant speaker per segment and stores it in the
   `speaker` column on `transcript_segment`.

Both integrations are **non-fatal**: if either fails the podcast still processes normally
and the failure is logged as a warning.

## Commits in this branch

| SHA | Description |
|-----|-------------|
| `418b157` | DB models: nullable `speaker` column on `transcript_segment`; new `audio_segment` table |
| `5ba6d53` | `WhisperXTranscriber` — diarize + align request, dominant-speaker extraction |
| `9578d1d` | `ina_client.py` — standalone INA API client returning `AudioSegmentResult` dataclasses |
| `7902fa6` | Writer actions: `replace_audio_segments_action`; propagate `speaker` through `replace_transcription_action` |
| `6f2ec07` | `podcast_processor.py` — background-thread INA analysis alongside transcription |
| `a8c22b0` | Config store: `WHISPER_TYPE=whisperx`, `INA_ENABLED`, `INA_BASE_URL` env-var overrides |
| `599c0cb` | Alembic migration: `speaker` column + `audio_segment` table |
| `747ed57` | Stats API + UI: speaker column in transcript table; Audio Segments tab |
| `e660bc2` | Stats UI: inline non-speech markers in transcript view; Speakers tab (time, words, %) |
| `46e121a` | Stats UI: horizontally scrollable tab bar on narrow screens |
| `af2b4c7` | Fix: always emit `speaker` column in transcript segment INSERT |

## Configuration

All settings are controlled via environment variables (authoritative over DB values):

| Env var | Default | Purpose |
|---------|---------|---------|
| `WHISPER_TYPE` | `remote` | Set to `whisperx` to enable diarization |
| `INA_ENABLED` | `false` | Set to `true` to enable INA segmentation |
| `INA_BASE_URL` | — | Base URL for the INA API service |
| `WHISPERX_BASE_URL` | — | Base URL for the WhisperX API service |

## Data model

```
transcript_segment
  + speaker  TEXT NULLABLE   -- "SPEAKER_00", "SPEAKER_01", … from WhisperX

audio_segment (new table)
  id          INTEGER PK
  post_id     INTEGER FK → post
  model_call_id INTEGER FK → model_call
  label       TEXT            -- "speech" | "music" | "noEnergy" | …
  start_time  FLOAT           -- seconds
  end_time    FLOAT           -- seconds
```

## Stats UI additions

- **Transcript tab** — each row gains a colour-coded speaker badge when diarization is active.
- **Audio Segments tab** — raw INA classification results with label, start, and end times.
- **Speakers tab** — per-speaker summary: segment count, total speaking time, word count,
  and percentage of episode.
- Non-speech segments (music, noise) are also merged inline with the transcript rows,
  sorted by `start_time`, so you can see music stingers around ad blocks at a glance.

## Running locally

1. Set `WHISPER_TYPE=whisperx` and `WHISPERX_BASE_URL=<your-whisperx-url>` to enable diarization.
2. Set `INA_ENABLED=true` and `INA_BASE_URL=<your-ina-url>` to enable speech segmentation.
3. Both services must expose the standard INA / WhisperX HTTP API contracts.
4. Re-process any episode from the UI to trigger both analyses; results appear immediately
   in the Stats drawer.
