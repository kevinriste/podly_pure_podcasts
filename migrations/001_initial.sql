-- Initial schema for Podly Rust backend
-- Mirrors Python/SQLAlchemy schema for migration compatibility

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Core entities

CREATE TABLE IF NOT EXISTS feed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alt_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    author TEXT,
    rss_url TEXT UNIQUE NOT NULL,
    image_url TEXT,
    ad_detection_strategy TEXT NOT NULL DEFAULT 'inherit',
    chapter_filter_strings TEXT,
    auto_whitelist_new_episodes_override INTEGER
);

CREATE TABLE IF NOT EXISTS post (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id INTEGER NOT NULL REFERENCES feed(id),
    guid TEXT UNIQUE NOT NULL,
    download_url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    unprocessed_audio_path TEXT,
    processed_audio_path TEXT,
    description TEXT,
    release_date TEXT,
    duration INTEGER,
    whitelisted INTEGER NOT NULL DEFAULT 0,
    image_url TEXT,
    download_count INTEGER DEFAULT 0,
    chapter_data TEXT,
    refined_ad_boundaries TEXT,
    refined_ad_boundaries_updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_post_feed_id ON post(feed_id);
CREATE INDEX IF NOT EXISTS ix_post_release_date ON post(release_date);

CREATE TABLE IF NOT EXISTS transcript_segment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL REFERENCES post(id),
    sequence_num INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_transcript_segment_post_id_sequence_num
    ON transcript_segment(post_id, sequence_num);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    feed_allowance INTEGER NOT NULL DEFAULT 0,
    feed_subscription_status TEXT NOT NULL DEFAULT 'inactive',
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    discord_id TEXT UNIQUE,
    discord_username TEXT,
    last_active TEXT,
    manual_feed_allowance INTEGER
);
CREATE INDEX IF NOT EXISTS ix_users_username ON users(username);

CREATE TABLE IF NOT EXISTS model_call (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER NOT NULL REFERENCES post(id),
    first_segment_sequence_num INTEGER NOT NULL,
    last_segment_sequence_num INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    retry_attempts INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_model_call_post_chunk_model
    ON model_call(post_id, first_segment_sequence_num, last_segment_sequence_num, model_name);

CREATE TABLE IF NOT EXISTS identification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_segment_id INTEGER NOT NULL REFERENCES transcript_segment(id),
    model_call_id INTEGER NOT NULL REFERENCES model_call(id),
    confidence REAL,
    label TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_identification_segment_call_label
    ON identification(transcript_segment_id, model_call_id, label);

CREATE TABLE IF NOT EXISTS jobs_manager_run (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    trigger TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    total_jobs INTEGER NOT NULL DEFAULT 0,
    queued_jobs INTEGER NOT NULL DEFAULT 0,
    running_jobs INTEGER NOT NULL DEFAULT 0,
    completed_jobs INTEGER NOT NULL DEFAULT 0,
    failed_jobs INTEGER NOT NULL DEFAULT 0,
    skipped_jobs INTEGER NOT NULL DEFAULT 0,
    context_json TEXT,
    counters_reset_at TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_manager_run_status ON jobs_manager_run(status);

CREATE TABLE IF NOT EXISTS processing_job (
    id TEXT PRIMARY KEY,
    jobs_manager_run_id TEXT REFERENCES jobs_manager_run(id),
    post_guid TEXT NOT NULL,
    status TEXT NOT NULL,
    current_step INTEGER DEFAULT 0,
    step_name TEXT,
    total_steps INTEGER DEFAULT 4,
    progress_percentage REAL DEFAULT 0.0,
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    scheduler_job_id TEXT,
    created_at TEXT,
    requested_by_user_id INTEGER REFERENCES users(id),
    billing_user_id INTEGER REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS ix_processing_job_post_guid ON processing_job(post_guid);
CREATE INDEX IF NOT EXISTS ix_processing_job_run_id ON processing_job(jobs_manager_run_id);
CREATE INDEX IF NOT EXISTS ix_processing_job_created_at ON processing_job(created_at);

CREATE TABLE IF NOT EXISTS feed_access_token (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id TEXT UNIQUE NOT NULL,
    token_hash TEXT NOT NULL,
    token_secret TEXT,
    feed_id INTEGER REFERENCES feed(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_feed_access_token_token_id ON feed_access_token(token_id);

CREATE TABLE IF NOT EXISTS feed_supporter (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feed_id INTEGER NOT NULL REFERENCES feed(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    UNIQUE(feed_id, user_id)
);

-- Settings tables (singletons, id=1)

CREATE TABLE IF NOT EXISTS llm_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    llm_api_key TEXT,
    llm_model TEXT NOT NULL DEFAULT 'groq/openai/gpt-oss-120b',
    oneshot_model TEXT,
    oneshot_max_chunk_duration_seconds INTEGER NOT NULL DEFAULT 7200,
    oneshot_chunk_overlap_seconds INTEGER NOT NULL DEFAULT 900,
    openai_base_url TEXT,
    openai_timeout INTEGER NOT NULL DEFAULT 300,
    openai_max_tokens INTEGER NOT NULL DEFAULT 4096,
    llm_max_concurrent_calls INTEGER NOT NULL DEFAULT 3,
    llm_max_retry_attempts INTEGER NOT NULL DEFAULT 5,
    llm_max_input_tokens_per_call INTEGER,
    llm_enable_token_rate_limiting INTEGER NOT NULL DEFAULT 0,
    llm_max_input_tokens_per_minute INTEGER,
    enable_boundary_refinement INTEGER NOT NULL DEFAULT 1,
    enable_word_level_boundary_refinder INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS whisper_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    whisper_type TEXT NOT NULL DEFAULT 'groq',
    local_model TEXT NOT NULL DEFAULT 'base.en',
    remote_model TEXT NOT NULL DEFAULT 'whisper-1',
    remote_api_key TEXT,
    remote_base_url TEXT NOT NULL DEFAULT 'https://api.openai.com/v1',
    remote_language TEXT NOT NULL DEFAULT 'en',
    remote_timeout_sec INTEGER NOT NULL DEFAULT 600,
    remote_chunksize_mb INTEGER NOT NULL DEFAULT 24,
    groq_api_key TEXT,
    groq_model TEXT NOT NULL DEFAULT 'whisper-large-v3-turbo',
    groq_language TEXT NOT NULL DEFAULT 'en',
    groq_max_retries INTEGER NOT NULL DEFAULT 3,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS processing_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    system_prompt_path TEXT NOT NULL DEFAULT 'src/system_prompt.txt',
    user_prompt_template_path TEXT NOT NULL DEFAULT 'src/user_prompt.jinja',
    num_segments_to_input_to_prompt INTEGER NOT NULL DEFAULT 60,
    max_overlap_segments INTEGER NOT NULL DEFAULT 30,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS output_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    fade_ms INTEGER NOT NULL DEFAULT 3000,
    min_ad_segement_separation_seconds INTEGER NOT NULL DEFAULT 60,
    min_ad_segment_length_seconds INTEGER NOT NULL DEFAULT 14,
    min_confidence REAL NOT NULL DEFAULT 0.8,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    background_update_interval_minute INTEGER,
    automatically_whitelist_new_episodes INTEGER NOT NULL DEFAULT 1,
    post_cleanup_retention_days INTEGER DEFAULT 5,
    number_of_episodes_to_whitelist_from_archive_of_new_feed INTEGER NOT NULL DEFAULT 1,
    ad_detection_strategy TEXT NOT NULL DEFAULT 'llm',
    enable_public_landing_page INTEGER NOT NULL DEFAULT 0,
    user_limit_total INTEGER,
    autoprocess_on_download INTEGER NOT NULL DEFAULT 0,
    env_config_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discord_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    client_id TEXT,
    client_secret TEXT,
    redirect_uri TEXT,
    guild_ids TEXT,
    allow_registration INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chapter_filter_settings (
    id INTEGER PRIMARY KEY DEFAULT 1,
    default_filter_strings TEXT NOT NULL DEFAULT 'sponsor,advertisement,ad break,promo,brought to you by',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
