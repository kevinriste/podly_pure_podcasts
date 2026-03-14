#[cfg(feature = "local-whisper")]
use std::path::Path;

#[cfg(feature = "local-whisper")]
use super::{Segment, Transcriber, TranscriptionError, TranscriptionResult};

#[cfg(feature = "local-whisper")]
pub struct LocalWhisperTranscriber {
    model_name: String,
    model_path: String,
}

#[cfg(feature = "local-whisper")]
impl LocalWhisperTranscriber {
    pub fn new(model_name: &str, model_path: &str) -> Self {
        Self {
            model_name: format!("local_{model_name}"),
            model_path: model_path.to_string(),
        }
    }
}

#[cfg(feature = "local-whisper")]
#[async_trait::async_trait]
impl Transcriber for LocalWhisperTranscriber {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn transcribe(
        &self,
        audio_file_path: &str,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        let model_path = self.model_path.clone();
        let audio_path = audio_file_path.to_string();

        tokio::task::spawn_blocking(move || run_whisper(&model_path, &audio_path))
            .await
            .map_err(|e| TranscriptionError::Whisper(format!("task join error: {e}")))?
    }
}

#[cfg(feature = "local-whisper")]
fn run_whisper(
    model_path: &str,
    audio_path: &str,
) -> Result<TranscriptionResult, TranscriptionError> {
    use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

    if !Path::new(model_path).exists() {
        return Err(TranscriptionError::Whisper(format!(
            "Model file not found: {model_path}"
        )));
    }

    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|e| TranscriptionError::Whisper(format!("failed to load model: {e}")))?;

    let audio_data = read_audio_to_f32(audio_path)?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    let mut state = ctx
        .create_state()
        .map_err(|e| TranscriptionError::Whisper(format!("failed to create state: {e}")))?;

    state
        .full(params, &audio_data)
        .map_err(|e| TranscriptionError::Whisper(format!("transcription failed: {e}")))?;

    let num_segments = state
        .full_n_segments()
        .map_err(|e| TranscriptionError::Whisper(format!("failed to get segment count: {e}")))?;

    let mut segments = Vec::new();
    let mut raw_segments = Vec::new();

    for i in 0..num_segments {
        let start = state
            .full_get_segment_t0(i)
            .map_err(|e| TranscriptionError::Whisper(format!("segment start: {e}")))?;
        let end = state
            .full_get_segment_t1(i)
            .map_err(|e| TranscriptionError::Whisper(format!("segment end: {e}")))?;
        let text = state
            .full_get_segment_text(i)
            .map_err(|e| TranscriptionError::Whisper(format!("segment text: {e}")))?;

        let start_sec = start as f64 / 100.0;
        let end_sec = end as f64 / 100.0;

        raw_segments.push(serde_json::json!({
            "start": start_sec,
            "end": end_sec,
            "text": text,
        }));

        segments.push(Segment {
            start: start_sec,
            end: end_sec,
            text,
        });
    }

    Ok(TranscriptionResult {
        segments,
        raw_response: serde_json::to_string_pretty(&raw_segments).unwrap_or_default(),
    })
}

#[cfg(feature = "local-whisper")]
fn read_audio_to_f32(audio_path: &str) -> Result<Vec<f32>, TranscriptionError> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            audio_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()?;

    if !output.status.success() {
        return Err(TranscriptionError::Whisper(
            "ffmpeg audio conversion failed".to_string(),
        ));
    }

    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(samples)
}
