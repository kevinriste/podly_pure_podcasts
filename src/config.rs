use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub static_dir: PathBuf,
    pub secret_key: String,

    // Auth
    pub require_auth: bool,
    pub default_admin_username: Option<String>,
    pub default_admin_password: Option<String>,

    // Env-level overrides (override DB settings when set — 12-factor, never persisted)
    pub llm_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub llm_model: Option<String>,
    pub oneshot_model: Option<String>,
    pub oneshot_api_key: Option<String>,
    pub openai_base_url: Option<String>,
    pub gemini_api_key: Option<String>,

    // Whisper env overrides
    pub whisper_type: Option<String>,
    pub groq_api_key: Option<String>,
    pub groq_whisper_model: Option<String>,
    pub whisper_remote_api_key: Option<String>,
    pub whisper_remote_base_url: Option<String>,
    pub whisper_remote_model: Option<String>,
    pub whisper_local_model: Option<String>,
    pub groq_max_retries: Option<String>,

    // CORS
    pub cors_origins: Option<String>,

    // Scheduler
    pub disable_scheduler: bool,

    // Podcast Index API (for search)
    pub podcast_index_api_key: Option<String>,
    pub podcast_index_api_secret: Option<String>,

    // Discord OAuth (env overrides)
    pub discord_client_id: Option<String>,
    pub discord_client_secret: Option<String>,
    pub discord_redirect_uri: Option<String>,
    pub discord_guild_ids: Option<String>,
    pub discord_allow_registration: Option<bool>,

    // Stripe billing
    pub stripe_secret_key: Option<String>,
    pub stripe_product_id: Option<String>,
    pub stripe_webhook_secret: Option<String>,
    pub stripe_min_subscription_amount_cents: i64,

    // Base URL for RSS feed links (auto-detected from host/port if unset)
    pub base_url: Option<String>,

    // Developer mode
    pub developer_mode: bool,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            database_url: env::var("DATABASE_URL")
                .unwrap_or_else(|_| "sqlite:data/podly.db?mode=rwc".to_string()),
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            data_dir: PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "data".to_string())),
            static_dir: PathBuf::from(
                env::var("STATIC_DIR").unwrap_or_else(|_| "static".to_string()),
            ),
            secret_key: env::var("PODLY_SECRET_KEY")
                .or_else(|_| env::var("SECRET_KEY"))
                .unwrap_or_else(|_| "change-me-in-production".to_string()),

            require_auth: env::var("REQUIRE_AUTH")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            default_admin_username: env::var("PODLY_ADMIN_USERNAME")
                .or_else(|_| env::var("DEFAULT_ADMIN_USERNAME"))
                .ok(),
            default_admin_password: env::var("PODLY_ADMIN_PASSWORD")
                .or_else(|_| env::var("DEFAULT_ADMIN_PASSWORD"))
                .ok(),

            llm_api_key: env::var("LLM_API_KEY").ok(),
            openai_api_key: env::var("OPENAI_API_KEY").ok(),
            llm_model: env::var("LLM_MODEL").ok(),
            oneshot_model: env::var("ONESHOT_MODEL")
                .or_else(|_| env::var("LLM_ONESHOT_MODEL"))
                .ok(),
            oneshot_api_key: env::var("ONESHOT_API_KEY").ok(),
            openai_base_url: env::var("OPENAI_BASE_URL").ok(),
            gemini_api_key: env::var("GEMINI_API_KEY").ok(),

            whisper_type: env::var("WHISPER_TYPE").ok(),
            groq_api_key: env::var("GROQ_API_KEY").ok(),
            groq_whisper_model: env::var("GROQ_WHISPER_MODEL")
                .or_else(|_| env::var("WHISPER_GROQ_MODEL"))
                .ok(),
            whisper_remote_api_key: env::var("WHISPER_REMOTE_API_KEY").ok(),
            whisper_remote_base_url: env::var("WHISPER_REMOTE_BASE_URL").ok(),
            whisper_remote_model: env::var("WHISPER_REMOTE_MODEL").ok(),
            whisper_local_model: env::var("WHISPER_LOCAL_MODEL").ok(),
            groq_max_retries: env::var("GROQ_MAX_RETRIES").ok(),

            cors_origins: env::var("CORS_ORIGINS").ok(),

            disable_scheduler: env::var("PODLY_DISABLE_SCHEDULER")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),

            podcast_index_api_key: env::var("PODCAST_INDEX_API_KEY").ok(),
            podcast_index_api_secret: env::var("PODCAST_INDEX_API_SECRET").ok(),

            discord_client_id: env::var("DISCORD_CLIENT_ID").ok(),
            discord_client_secret: env::var("DISCORD_CLIENT_SECRET").ok(),
            discord_redirect_uri: env::var("DISCORD_REDIRECT_URI").ok(),
            discord_guild_ids: env::var("DISCORD_GUILD_IDS").ok(),
            discord_allow_registration: env::var("DISCORD_ALLOW_REGISTRATION")
                .ok()
                .map(|v| v == "true" || v == "1" || v == "yes"),

            stripe_secret_key: env::var("STRIPE_SECRET_KEY").ok(),
            stripe_product_id: env::var("STRIPE_PRODUCT_ID").ok(),
            stripe_webhook_secret: env::var("STRIPE_WEBHOOK_SECRET").ok(),
            stripe_min_subscription_amount_cents: env::var("STRIPE_MIN_SUBSCRIPTION_AMOUNT_CENTS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(100),

            base_url: env::var("BASE_URL")
                .or_else(|_| env::var("PODLY_BASE_URL"))
                .ok()
                .map(|u| u.trim_end_matches('/').to_string()),

            developer_mode: env::var("DEVELOPER_MODE")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
        }
    }
}
