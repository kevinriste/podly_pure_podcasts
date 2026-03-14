pub mod groq;
pub mod local;
pub mod remote;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Common interface for all transcription backends.
#[async_trait::async_trait]
pub trait Transcriber: Send + Sync {
    #[allow(dead_code)]
    fn model_name(&self) -> &str;
    async fn transcribe(
        &self,
        audio_file_path: &str,
    ) -> Result<TranscriptionResult, TranscriptionError>;
}

#[derive(Debug)]
pub struct TranscriptionResult {
    pub segments: Vec<Segment>,
    pub raw_response: String,
}

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum TranscriptionError {
    #[error("whisper error: {0}")]
    Whisper(String),
    #[error("api error: {0}")]
    Api(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
