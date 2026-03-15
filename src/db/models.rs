use serde::{Deserialize, Serialize};
use sqlx::FromRow;

// ── Core entities ──

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Feed {
    pub id: i64,
    pub alt_id: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub rss_url: String,
    pub image_url: Option<String>,
    pub ad_detection_strategy: String,
    pub chapter_filter_strings: Option<String>,
    pub auto_whitelist_new_episodes_override: Option<bool>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Post {
    pub id: i64,
    pub feed_id: i64,
    pub guid: String,
    pub download_url: String,
    pub title: String,
    pub unprocessed_audio_path: Option<String>,
    pub processed_audio_path: Option<String>,
    pub description: Option<String>,
    pub release_date: Option<String>,
    pub duration: Option<i64>,
    pub whitelisted: bool,
    pub image_url: Option<String>,
    pub download_count: Option<i64>,
    pub chapter_data: Option<String>,
    pub refined_ad_boundaries: Option<String>,
    pub refined_ad_boundaries_updated_at: Option<String>,
    /// Only populated for aggregate feed queries (joined from feed table)
    #[sqlx(default)]
    pub feed_title: Option<String>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub id: i64,
    pub post_id: i64,
    pub sequence_num: i64,
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub username: String,
    #[serde(skip_serializing)]
    pub password_hash: String,
    pub role: String,
    pub feed_allowance: i64,
    pub feed_subscription_status: String,
    pub stripe_customer_id: Option<String>,
    pub stripe_subscription_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub discord_id: Option<String>,
    pub discord_username: Option<String>,
    pub last_active: Option<String>,
    pub manual_feed_allowance: Option<i64>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct ModelCall {
    pub id: i64,
    pub post_id: i64,
    pub first_segment_sequence_num: i64,
    pub last_segment_sequence_num: i64,
    pub model_name: String,
    pub prompt: String,
    pub response: Option<String>,
    pub timestamp: String,
    pub status: String,
    pub error_message: Option<String>,
    pub retry_attempts: i64,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct Identification {
    pub id: i64,
    pub transcript_segment_id: i64,
    pub model_call_id: i64,
    pub confidence: Option<f64>,
    pub label: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct JobsManagerRun {
    pub id: String,
    pub status: String,
    pub trigger: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub total_jobs: i64,
    pub queued_jobs: i64,
    pub running_jobs: i64,
    pub completed_jobs: i64,
    pub failed_jobs: i64,
    pub skipped_jobs: i64,
    pub context_json: Option<String>,
    pub counters_reset_at: Option<String>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct ProcessingJob {
    pub id: String,
    pub jobs_manager_run_id: Option<String>,
    pub post_guid: String,
    pub status: String,
    pub current_step: Option<i64>,
    pub step_name: Option<String>,
    pub total_steps: Option<i64>,
    pub progress_percentage: Option<f64>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub error_message: Option<String>,
    pub scheduler_job_id: Option<String>,
    pub created_at: Option<String>,
    pub requested_by_user_id: Option<i64>,
    pub billing_user_id: Option<i64>,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct FeedAccessToken {
    pub id: i64,
    pub token_id: String,
    pub token_hash: String,
    pub token_secret: Option<String>,
    pub feed_id: Option<i64>,
    pub user_id: i64,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub revoked: bool,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct UserFeed {
    pub id: i64,
    pub feed_id: i64,
    pub user_id: i64,
    pub created_at: String,
}

// ── Settings singletons ──

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct LlmSettings {
    pub id: i64,
    pub llm_api_key: Option<String>,
    pub llm_model: String,
    pub oneshot_model: Option<String>,
    pub oneshot_max_chunk_duration_seconds: i64,
    pub oneshot_chunk_overlap_seconds: i64,
    pub openai_base_url: Option<String>,
    pub openai_timeout: i64,
    pub openai_max_tokens: i64,
    pub llm_max_concurrent_calls: i64,
    pub llm_max_retry_attempts: i64,
    pub llm_max_input_tokens_per_call: Option<i64>,
    pub llm_enable_token_rate_limiting: bool,
    pub llm_max_input_tokens_per_minute: Option<i64>,
    pub enable_boundary_refinement: bool,
    pub enable_word_level_boundary_refinder: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct WhisperSettings {
    pub id: i64,
    pub whisper_type: String,
    pub local_model: String,
    pub remote_model: String,
    pub remote_api_key: Option<String>,
    pub remote_base_url: String,
    pub remote_language: String,
    pub remote_timeout_sec: i64,
    pub remote_chunksize_mb: i64,
    pub groq_api_key: Option<String>,
    pub groq_model: String,
    pub groq_language: String,
    pub groq_max_retries: i64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct ProcessingSettings {
    pub id: i64,
    pub system_prompt_path: String,
    pub user_prompt_template_path: String,
    pub num_segments_to_input_to_prompt: i64,
    pub max_overlap_segments: i64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct OutputSettings {
    pub id: i64,
    pub fade_ms: i64,
    #[sqlx(rename = "min_ad_segement_separation_seconds")]
    pub min_ad_segment_separation_seconds: i64,
    pub min_ad_segment_length_seconds: i64,
    pub min_confidence: f64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct AppSettings {
    pub id: i64,
    pub background_update_interval_minute: Option<i64>,
    pub automatically_whitelist_new_episodes: bool,
    pub post_cleanup_retention_days: Option<i64>,
    pub number_of_episodes_to_whitelist_from_archive_of_new_feed: i64,
    pub ad_detection_strategy: String,
    pub enable_public_landing_page: bool,
    pub user_limit_total: Option<i64>,
    pub autoprocess_on_download: bool,
    pub env_config_hash: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct DiscordSettings {
    pub id: i64,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub redirect_uri: Option<String>,
    pub guild_ids: Option<String>,
    pub allow_registration: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct ChapterFilterSettings {
    pub id: i64,
    pub default_filter_strings: String,
    pub created_at: String,
    pub updated_at: String,
}
