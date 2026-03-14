use std::path::Path;

use super::{Segment, Transcriber, TranscriptionError, TranscriptionResult};

pub struct GroqWhisperTranscriber {
    model: String,
    api_key: String,
    language: String,
    max_retries: u32,
}

impl GroqWhisperTranscriber {
    pub fn new(model: &str, api_key: &str, language: &str, max_retries: u32) -> Self {
        Self {
            model: model.to_string(),
            api_key: api_key.to_string(),
            language: language.to_string(),
            max_retries,
        }
    }
}

#[async_trait::async_trait]
impl Transcriber for GroqWhisperTranscriber {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn transcribe(
        &self,
        audio_file_path: &str,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        // Groq has a ~25MB file limit; split larger files into chunks
        let file_size = tokio::fs::metadata(audio_file_path)
            .await
            .map_err(|e| TranscriptionError::Io(e))?
            .len();

        let max_chunk_bytes: u64 = 24 * 1024 * 1024; // 24MB

        if file_size <= max_chunk_bytes {
            // Single request
            return self.transcribe_single(audio_file_path).await;
        }

        // Split into chunks and transcribe each
        let tmp_dir = tempfile::tempdir()
            .map_err(|e| TranscriptionError::Io(e))?;

        let chunks = crate::audio::split_audio(
            Path::new(audio_file_path),
            tmp_dir.path(),
            max_chunk_bytes as usize,
        )
        .await
        .map_err(|e| TranscriptionError::Api(e.to_string()))?;

        let mut all_segments = Vec::new();
        let mut all_raw: Vec<serde_json::Value> = Vec::new();

        for (chunk_path, offset_ms) in &chunks {
            let chunk_path_str = chunk_path.to_str().unwrap_or("");
            let result = self.transcribe_single(chunk_path_str).await?;

            let offset_sec = *offset_ms as f64 / 1000.0;
            for mut seg in result.segments {
                seg.start += offset_sec;
                seg.end += offset_sec;
                all_segments.push(seg);
            }
            if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&result.raw_response) {
                all_raw.push(raw);
            }
        }

        Ok(TranscriptionResult {
            segments: all_segments,
            raw_response: serde_json::to_string(&all_raw).unwrap_or_default(),
        })
    }
}

impl GroqWhisperTranscriber {
    async fn transcribe_single(
        &self,
        audio_path: &str,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| TranscriptionError::Api(e.to_string()))?;

        let file_bytes = tokio::fs::read(audio_path)
            .await
            .map_err(|e| TranscriptionError::Io(e))?;

        let filename = Path::new(audio_path)
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("audio.mp3")
            .to_string();

        for attempt in 0..=self.max_retries {
            let file_part = reqwest::multipart::Part::bytes(file_bytes.clone())
                .file_name(filename.clone())
                .mime_str("audio/mpeg")
                .unwrap();

            let form = reqwest::multipart::Form::new()
                .part("file", file_part)
                .text("model", self.model.clone())
                .text("language", self.language.clone())
                .text("response_format", "verbose_json")
                .text("timestamp_granularities[]", "segment");

            let resp = client
                .post("https://api.groq.com/openai/v1/audio/transcriptions")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .multipart(form)
                .send()
                .await;

            match resp {
                Ok(r) if r.status().is_success() => {
                    let raw_text = r.text().await.map_err(|e| TranscriptionError::Api(e.to_string()))?;
                    let parsed: serde_json::Value = serde_json::from_str(&raw_text)
                        .map_err(|e| TranscriptionError::Api(format!("parse error: {e}")))?;

                    let segments = parse_whisper_segments(&parsed);
                    return Ok(TranscriptionResult {
                        segments,
                        raw_response: raw_text,
                    });
                }
                Ok(r) if r.status().as_u16() == 429 => {
                    let wait = std::time::Duration::from_secs(60 * 2u64.pow(attempt));
                    tracing::warn!("Groq rate limited (attempt {}), waiting {}s", attempt + 1, wait.as_secs());
                    tokio::time::sleep(wait).await;
                }
                Ok(r) => {
                    let status = r.status();
                    let body = r.text().await.unwrap_or_default();
                    if attempt < self.max_retries {
                        let wait = std::time::Duration::from_secs(2u64.pow(attempt));
                        tracing::warn!("Groq error {status}: {body}");
                        tokio::time::sleep(wait).await;
                    } else {
                        return Err(TranscriptionError::Api(format!("Groq error {status}: {body}")));
                    }
                }
                Err(e) => {
                    if attempt < self.max_retries {
                        tracing::warn!("Groq request error: {e}");
                        tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt))).await;
                    } else {
                        return Err(TranscriptionError::Api(e.to_string()));
                    }
                }
            }
        }

        Err(TranscriptionError::Api("max retries exceeded".into()))
    }
}

/// Parse whisper verbose_json response into segments.
fn parse_whisper_segments(json: &serde_json::Value) -> Vec<Segment> {
    let empty = vec![];
    let segments_arr = json["segments"].as_array().unwrap_or(&empty);

    segments_arr
        .iter()
        .filter_map(|seg| {
            let start = seg["start"].as_f64()?;
            let end = seg["end"].as_f64()?;
            let text = seg["text"].as_str()?.trim().to_string();
            if text.is_empty() {
                return None;
            }
            Some(Segment { start, end, text })
        })
        .collect()
}
