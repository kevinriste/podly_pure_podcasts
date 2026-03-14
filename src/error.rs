use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum AppError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),

    #[error("not found")]
    NotFound,

    #[error("unauthorized: {0}")]
    Unauthorized(String),

    #[error("forbidden")]
    Forbidden,

    #[error("bad request: {0}")]
    BadRequest(String),

    #[error("conflict: {0}")]
    Conflict(String),

    #[error("payment required: {0}")]
    PaymentRequired(String),

    #[error("too many requests")]
    TooManyRequests { retry_after: u64 },

    #[error("not implemented")]
    NotImplemented,

    #[error("llm error: {0}")]
    Llm(String),

    #[error("transcription error: {0}")]
    Transcription(String),

    #[error("audio processing error: {0}")]
    Audio(String),

    #[error("internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::Db(e) => {
                tracing::error!("Database error: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "database error".to_string(),
                )
            }
            AppError::NotFound => (StatusCode::NOT_FOUND, "not found".to_string()),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            AppError::Forbidden => (StatusCode::FORBIDDEN, "forbidden".to_string()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, msg.clone()),
            AppError::PaymentRequired(msg) => (StatusCode::PAYMENT_REQUIRED, msg.clone()),
            AppError::TooManyRequests { retry_after } => {
                let body = json!({ "error": "too many requests", "retry_after": retry_after });
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    [("Retry-After", retry_after.to_string())],
                    axum::Json(body),
                )
                    .into_response();
            }
            AppError::NotImplemented => {
                (StatusCode::NOT_IMPLEMENTED, "not implemented".to_string())
            }
            AppError::Llm(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Transcription(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Audio(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Internal(e) => {
                tracing::error!("Internal error: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal error".to_string(),
                )
            }
        };

        let body = json!({ "error": message });
        (status, axum::Json(body)).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
