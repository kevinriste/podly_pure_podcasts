use std::path::Path;

use super::{Segment, Transcriber, TranscriptionError, TranscriptionResult};

pub struct RemoteWhisperTranscriber {
    model: String,
    base_url: String,
    api_key: String,
    language: String,
    timeout_sec: u64,
    chunksize_mb: usize,
}

impl RemoteWhisperTranscriber {
    pub fn new(
        model: &str,
        base_url: &str,
        api_key: &str,
        language: &str,
        timeout_sec: u64,
        chunksize_mb: usize,
    ) -> Self {
        Self {
            model: model.to_string(),
            base_url: base_url.to_string(),
            api_key: api_key.to_string(),
            language: language.to_string(),
            timeout_sec,
            chunksize_mb,
        }
    }
}

#[async_trait::async_trait]
impl Transcriber for RemoteWhisperTranscriber {
    fn model_name(&self) -> &str {
        &self.model
    }

    async fn transcribe(
        &self,
        audio_file_path: &str,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let file_size = tokio::fs::metadata(audio_file_path)
            .await
            .map_err(|e| TranscriptionError::Io(e))?
            .len();

        let max_chunk_bytes = (self.chunksize_mb * 1024 * 1024) as u64;

        if file_size <= max_chunk_bytes {
            return self.transcribe_single(audio_file_path).await;
        }

        // Split and transcribe chunks
        let tmp_dir = tempfile::tempdir().map_err(|e| TranscriptionError::Io(e))?;

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
            let result = self
                .transcribe_single(chunk_path.to_str().unwrap_or(""))
                .await?;

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

impl RemoteWhisperTranscriber {
    async fn transcribe_single(
        &self,
        audio_path: &str,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout_sec))
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

        let file_part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(filename)
            .mime_str("audio/mpeg")
            .unwrap();

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.model.clone())
            .text("language", self.language.clone())
            .text("response_format", "verbose_json")
            .text("timestamp_granularities[]", "segment");

        let url = format!("{}/audio/transcriptions", self.base_url.trim_end_matches('/'));

        let resp = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await
            .map_err(|e| TranscriptionError::Api(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(TranscriptionError::Api(format!(
                "Remote whisper error {status}: {body}"
            )));
        }

        let raw_text = resp
            .text()
            .await
            .map_err(|e| TranscriptionError::Api(e.to_string()))?;

        let parsed: serde_json::Value = serde_json::from_str(&raw_text)
            .map_err(|e| TranscriptionError::Api(format!("parse error: {e}")))?;

        let segments = parse_whisper_segments(&parsed);

        Ok(TranscriptionResult {
            segments,
            raw_response: raw_text,
        })
    }
}

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
