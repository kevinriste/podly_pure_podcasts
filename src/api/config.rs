use axum::extract::State;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde_json::{json, Value};

use crate::auth::middleware::require_admin_user;
use crate::auth::AuthenticatedUser;
use crate::config::AppConfig;
use crate::db::queries;
use crate::error::{AppError, AppResult};
use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/config", get(get_config).put(update_config))
        .route("/api/config/test-llm", post(test_llm))
        .route("/api/config/test-oneshot", post(test_oneshot))
        .route("/api/config/test-whisper", post(test_whisper))
        .route(
            "/api/config/whisper-capabilities",
            get(whisper_capabilities),
        )
        .route("/api/config/api_configured_check", get(api_configured))
        .route("/api/landing/status", get(landing_status))
}

/// Mask an API key for display: show first 4 and last 4 chars.
fn mask(val: &Option<String>) -> Option<String> {
    val.as_ref().and_then(|s| {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        if s.len() <= 8 {
            return Some("****".into());
        }
        Some(format!("{}****{}", &s[..4], &s[s.len() - 4..]))
    })
}

/// Build the `env_overrides` metadata map following PR #196's 12-factor model.
/// Each entry has `env_var`, `read_only: true`, and optionally masked value info.
fn build_env_override_metadata(config: &AppConfig, data: &Value) -> Value {
    let mut overrides = serde_json::Map::new();

    // Helper to register an override
    let mut register = |path: &str, env_var: &str, value: &Option<String>, secret: bool| {
        if let Some(val) = value {
            if val.is_empty() {
                return;
            }
            let mut entry = serde_json::Map::new();
            entry.insert("env_var".into(), json!(env_var));
            entry.insert("read_only".into(), json!(true));
            if secret {
                entry.insert("is_secret".into(), json!(true));
                entry.insert("value_preview".into(), json!(mask(value)));
            } else {
                entry.insert("value".into(), json!(val));
            }
            overrides.insert(path.into(), Value::Object(entry));
        }
    };

    // LLM overrides
    let llm_api_key = config
        .llm_api_key
        .clone()
        .or_else(|| config.openai_api_key.clone())
        .or_else(|| config.groq_api_key.clone());
    register("llm.llm_api_key", "LLM_API_KEY", &llm_api_key, true);
    register(
        "llm.openai_base_url",
        "OPENAI_BASE_URL",
        &config.openai_base_url,
        false,
    );
    register("llm.llm_model", "LLM_MODEL", &config.llm_model, false);

    // Whisper overrides
    register(
        "whisper.whisper_type",
        "WHISPER_TYPE",
        &config.whisper_type,
        false,
    );

    // Whisper API key — depends on type
    let whisper_type = config.whisper_type.as_deref().unwrap_or(
        data.get("whisper")
            .and_then(|w| w.get("whisper_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("groq"),
    );

    match whisper_type {
        "remote" => {
            let remote_key = config
                .whisper_remote_api_key
                .clone()
                .or_else(|| config.openai_api_key.clone());
            register("whisper.api_key", "WHISPER_REMOTE_API_KEY", &remote_key, true);
            let remote_base = config
                .whisper_remote_base_url
                .clone()
                .or_else(|| config.openai_base_url.clone());
            register(
                "whisper.base_url",
                "WHISPER_REMOTE_BASE_URL",
                &remote_base,
                false,
            );
            register(
                "whisper.model",
                "WHISPER_REMOTE_MODEL",
                &config.whisper_remote_model,
                false,
            );
        }
        "groq" => {
            register("whisper.api_key", "GROQ_API_KEY", &config.groq_api_key, true);
            register(
                "whisper.model",
                "GROQ_WHISPER_MODEL",
                &config.groq_whisper_model,
                false,
            );
        }
        "local" => {
            register(
                "whisper.model",
                "WHISPER_LOCAL_MODEL",
                &config.whisper_local_model,
                false,
            );
        }
        _ => {}
    }

    Value::Object(overrides)
}

/// Return the set of field paths overridden by env vars.
fn get_env_overridden_fields(config: &AppConfig) -> Vec<&'static str> {
    let mut fields = Vec::new();

    if config
        .llm_api_key
        .is_some()
        || config.openai_api_key.is_some()
        || config.groq_api_key.is_some()
    {
        fields.push("llm.llm_api_key");
    }
    if config.openai_base_url.is_some() {
        fields.push("llm.openai_base_url");
    }
    if config.llm_model.is_some() {
        fields.push("llm.llm_model");
    }
    if config.whisper_type.is_some() {
        fields.push("whisper.whisper_type");
    }
    if config.whisper_remote_api_key.is_some() || config.openai_api_key.is_some() {
        fields.push("whisper.api_key");
    }
    if config.whisper_remote_base_url.is_some() || config.openai_base_url.is_some() {
        fields.push("whisper.base_url");
    }
    if config.whisper_remote_model.is_some()
        || config.groq_whisper_model.is_some()
        || config.whisper_local_model.is_some()
    {
        fields.push("whisper.model");
    }
    if config.groq_api_key.is_some() {
        fields.push("whisper.api_key");
    }

    fields
}

async fn get_config(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let llm = queries::get_llm_settings(&state.db).await?;
    let whisper = queries::get_whisper_settings(&state.db).await?;
    let processing = queries::get_processing_settings(&state.db).await?;
    let output = queries::get_output_settings(&state.db).await?;
    let app = queries::get_app_settings(&state.db).await?;
    let chapter = queries::get_chapter_filter_settings(&state.db).await?;

    // Build base data from DB
    let mut data = json!({
        "llm": {
            "llm_api_key": "",
            "llm_api_key_preview": mask(&llm.llm_api_key),
            "llm_model": &llm.llm_model,
            "oneshot_model": llm.oneshot_model.as_deref(),
            "openai_base_url": llm.openai_base_url.as_deref(),
            "openai_timeout": llm.openai_timeout,
            "openai_max_tokens": llm.openai_max_tokens,
            "llm_max_concurrent_calls": llm.llm_max_concurrent_calls,
            "llm_max_retry_attempts": llm.llm_max_retry_attempts,
            "llm_max_input_tokens_per_call": llm.llm_max_input_tokens_per_call,
            "llm_enable_token_rate_limiting": llm.llm_enable_token_rate_limiting,
            "llm_max_input_tokens_per_minute": llm.llm_max_input_tokens_per_minute,
            "enable_boundary_refinement": llm.enable_boundary_refinement,
            "enable_word_level_boundary_refinder": llm.enable_word_level_boundary_refinder,
            "oneshot_max_chunk_duration_seconds": llm.oneshot_max_chunk_duration_seconds,
            "oneshot_chunk_overlap_seconds": llm.oneshot_chunk_overlap_seconds,
        },
        "whisper": {
            "whisper_type": &whisper.whisper_type,
            "local_model": &whisper.local_model,
            "remote_model": &whisper.remote_model,
            "remote_api_key": "",
            "remote_api_key_preview": mask(&whisper.remote_api_key),
            "remote_base_url": &whisper.remote_base_url,
            "remote_language": &whisper.remote_language,
            "remote_timeout_sec": whisper.remote_timeout_sec,
            "remote_chunksize_mb": whisper.remote_chunksize_mb,
            "groq_api_key": "",
            "groq_api_key_preview": mask(&state.config.groq_api_key.clone().or(whisper.groq_api_key.clone())),
            "groq_model": &whisper.groq_model,
            "groq_language": &whisper.groq_language,
            "groq_max_retries": whisper.groq_max_retries,
        },
        "processing": {
            "system_prompt_path": &processing.system_prompt_path,
            "user_prompt_template_path": &processing.user_prompt_template_path,
            "num_segments_to_input_to_prompt": processing.num_segments_to_input_to_prompt,
        },
        "output": {
            "fade_ms": output.fade_ms,
            "min_ad_segment_separation_seconds": output.min_ad_segment_separation_seconds,
            "min_ad_segment_length_seconds": output.min_ad_segment_length_seconds,
            "min_confidence": output.min_confidence,
        },
        "app": {
            "background_update_interval_minute": app.background_update_interval_minute,
            "automatically_whitelist_new_episodes": app.automatically_whitelist_new_episodes,
            "post_cleanup_retention_days": app.post_cleanup_retention_days,
            "number_of_episodes_to_whitelist_from_archive_of_new_feed": app.number_of_episodes_to_whitelist_from_archive_of_new_feed,
            "ad_detection_strategy": &app.ad_detection_strategy,
            "enable_public_landing_page": app.enable_public_landing_page,
            "user_limit_total": app.user_limit_total,
            "autoprocess_on_download": app.autoprocess_on_download,
        },
        "chapter_filter": {
            "default_filter_strings": &chapter.default_filter_strings,
        },
    });

    // Apply env var overlays on top of DB values (runtime only, never persisted)
    if let Some(llm_section) = data.get_mut("llm").and_then(|v| v.as_object_mut()) {
        if let Some(key) = state
            .config
            .llm_api_key
            .as_ref()
            .or(state.config.openai_api_key.as_ref())
            .or(state.config.groq_api_key.as_ref())
        {
            llm_section.insert("llm_api_key_preview".into(), json!(mask(&Some(key.clone()))));
        }
        if let Some(model) = &state.config.llm_model {
            llm_section.insert("llm_model".into(), json!(model));
        }
        if let Some(url) = &state.config.openai_base_url {
            llm_section.insert("openai_base_url".into(), json!(url));
        }
    }

    if let Some(w_section) = data.get_mut("whisper").and_then(|v| v.as_object_mut()) {
        if let Some(wt) = &state.config.whisper_type {
            w_section.insert("whisper_type".into(), json!(wt));
        }
        if let Some(key) = &state.config.groq_api_key {
            w_section.insert("groq_api_key_preview".into(), json!(mask(&Some(key.clone()))));
        }
        if let Some(key) = state
            .config
            .whisper_remote_api_key
            .as_ref()
            .or(state.config.openai_api_key.as_ref())
        {
            w_section.insert(
                "remote_api_key_preview".into(),
                json!(mask(&Some(key.clone()))),
            );
        }
    }

    // Build env_overrides metadata (tells frontend which fields are read-only)
    let env_overrides = build_env_override_metadata(&state.config, &data);

    // Add read_only_fields list for convenience
    let read_only_fields: Vec<&str> = get_env_overridden_fields(&state.config);

    // Insert read_only_fields into the config object
    data.as_object_mut()
        .unwrap()
        .insert("read_only_fields".into(), json!(read_only_fields));

    // Wrap under { config: ..., env_overrides: ... } to match Python response shape
    let result = json!({
        "config": data,
        "env_overrides": env_overrides,
    });

    Ok(Json(result))
}

async fn update_config(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(mut body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    // Strip env-overridden fields — env vars are authoritative (PR #196 model)
    let overridden = get_env_overridden_fields(&state.config);
    let mut stripped: Vec<String> = Vec::new();

    if let Some(llm) = body.get_mut("llm").and_then(|v| v.as_object_mut()) {
        if overridden.contains(&"llm.llm_api_key") {
            if llm.remove("llm_api_key").is_some() {
                stripped.push("llm.llm_api_key".into());
            }
        }
        if overridden.contains(&"llm.openai_base_url") {
            if llm.remove("openai_base_url").is_some() {
                stripped.push("llm.openai_base_url".into());
            }
        }
        if overridden.contains(&"llm.llm_model") {
            if llm.remove("llm_model").is_some() {
                stripped.push("llm.llm_model".into());
            }
        }
    }

    if let Some(w) = body.get_mut("whisper").and_then(|v| v.as_object_mut()) {
        if overridden.contains(&"whisper.whisper_type") {
            if w.remove("whisper_type").is_some() {
                stripped.push("whisper.whisper_type".into());
            }
        }
        if overridden.contains(&"whisper.api_key") {
            // Strip all whisper API key fields
            for key in &["api_key", "remote_api_key", "groq_api_key"] {
                if w.remove(*key).is_some() {
                    stripped.push(format!("whisper.{key}"));
                }
            }
        }
        if overridden.contains(&"whisper.base_url") {
            if w.remove("base_url").is_some() || w.remove("remote_base_url").is_some() {
                stripped.push("whisper.base_url".into());
            }
        }
        if overridden.contains(&"whisper.model") {
            for key in &["model", "remote_model", "groq_model", "local_model"] {
                if w.remove(*key).is_some() {
                    stripped.push(format!("whisper.{key}"));
                }
            }
        }
    }

    if !stripped.is_empty() {
        tracing::info!(
            "Stripped env-overridden fields from config update: {}",
            stripped.join(", ")
        );
    }

    let now = chrono::Utc::now().to_rfc3339();

    // Update LLM settings
    if let Some(llm) = body.get("llm") {
        if let Some(key) = llm.get("llm_api_key").and_then(|v| v.as_str()) {
            if !key.contains("****") && !key.is_empty() {
                sqlx::query("UPDATE llm_settings SET llm_api_key = ?, updated_at = ? WHERE id = 1")
                    .bind(key)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
            }
        }
        if let Some(model) = llm.get("llm_model").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE llm_settings SET llm_model = ?, updated_at = ? WHERE id = 1")
                .bind(model)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("openai_base_url").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE llm_settings SET openai_base_url = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = llm.get("openai_timeout").and_then(|v| v.as_i64()) {
            sqlx::query(
                "UPDATE llm_settings SET openai_timeout = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = llm.get("openai_max_tokens").and_then(|v| v.as_i64()) {
            sqlx::query(
                "UPDATE llm_settings SET openai_max_tokens = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = llm.get("llm_max_concurrent_calls").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE llm_settings SET llm_max_concurrent_calls = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("llm_max_retry_attempts").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE llm_settings SET llm_max_retry_attempts = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("enable_boundary_refinement").and_then(|v| v.as_bool()) {
            sqlx::query("UPDATE llm_settings SET enable_boundary_refinement = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("oneshot_model").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE llm_settings SET oneshot_model = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
    }

    // Update whisper settings
    if let Some(w) = body.get("whisper") {
        if let Some(v) = w.get("whisper_type").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE whisper_settings SET whisper_type = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = w.get("remote_model").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE whisper_settings SET remote_model = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = w.get("remote_api_key").and_then(|v| v.as_str()) {
            if !v.contains("****") && !v.is_empty() {
                sqlx::query("UPDATE whisper_settings SET remote_api_key = ?, updated_at = ? WHERE id = 1")
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
            }
        }
        if let Some(v) = w.get("remote_base_url").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE whisper_settings SET remote_base_url = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = w.get("groq_api_key").and_then(|v| v.as_str()) {
            if !v.contains("****") && !v.is_empty() {
                sqlx::query("UPDATE whisper_settings SET groq_api_key = ?, updated_at = ? WHERE id = 1")
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
            }
        }
        if let Some(v) = w.get("groq_model").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE whisper_settings SET groq_model = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = w.get("remote_language").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE whisper_settings SET remote_language = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = w.get("groq_language").and_then(|v| v.as_str()) {
            sqlx::query(
                "UPDATE whisper_settings SET groq_language = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
    }

    // Update output settings
    if let Some(o) = body.get("output") {
        if let Some(v) = o.get("fade_ms").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE output_settings SET fade_ms = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = o
            .get("min_ad_segment_separation_seconds")
            .and_then(|v| v.as_i64())
        {
            sqlx::query("UPDATE output_settings SET min_ad_segement_separation_seconds = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = o
            .get("min_ad_segment_length_seconds")
            .and_then(|v| v.as_i64())
        {
            sqlx::query("UPDATE output_settings SET min_ad_segment_length_seconds = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = o.get("min_confidence").and_then(|v| v.as_f64()) {
            sqlx::query(
                "UPDATE output_settings SET min_confidence = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
    }

    // Update app settings
    if let Some(a) = body.get("app") {
        if let Some(v) = a
            .get("automatically_whitelist_new_episodes")
            .and_then(|v| v.as_bool())
        {
            sqlx::query("UPDATE app_settings SET automatically_whitelist_new_episodes = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a
            .get("background_update_interval_minute")
            .and_then(|v| v.as_i64())
        {
            sqlx::query("UPDATE app_settings SET background_update_interval_minute = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a.get("post_cleanup_retention_days") {
            let val = if v.is_null() { None } else { v.as_i64() };
            sqlx::query("UPDATE app_settings SET post_cleanup_retention_days = ?, updated_at = ? WHERE id = 1")
                .bind(val)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a
            .get("number_of_episodes_to_whitelist_from_archive_of_new_feed")
            .and_then(|v| v.as_i64())
        {
            sqlx::query("UPDATE app_settings SET number_of_episodes_to_whitelist_from_archive_of_new_feed = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a.get("ad_detection_strategy").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE app_settings SET ad_detection_strategy = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a
            .get("enable_public_landing_page")
            .and_then(|v| v.as_bool())
        {
            sqlx::query("UPDATE app_settings SET enable_public_landing_page = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a.get("user_limit_total").and_then(|v| v.as_i64()) {
            sqlx::query(
                "UPDATE app_settings SET user_limit_total = ?, updated_at = ? WHERE id = 1",
            )
            .bind(v)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }
        if let Some(v) = a
            .get("autoprocess_on_download")
            .and_then(|v| v.as_bool())
        {
            sqlx::query("UPDATE app_settings SET autoprocess_on_download = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
    }

    // Update chapter filter settings
    if let Some(c) = body.get("chapter_filter") {
        if let Some(v) = c.get("default_filter_strings").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE chapter_filter_settings SET default_filter_strings = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
    }

    Ok(Json(json!({"status": "ok"})))
}

async fn test_llm(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let api_key = body
        .get("api_key")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-4");
    let base_url = body
        .get("base_url")
        .and_then(|v| v.as_str())
        .unwrap_or("https://api.openai.com/v1");

    let client = reqwest::Client::new();
    let resp = client
        .post(&format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&json!({
            "model": model,
            "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
            "max_tokens": 10,
        }))
        .send()
        .await
        .map_err(|e| AppError::Llm(format!("Connection error: {e}")))?;

    if resp.status().is_success() {
        Ok(Json(
            json!({"status": "ok", "message": "LLM connection successful."}),
        ))
    } else {
        let status = resp.status().as_u16();
        let body_text = resp.text().await.unwrap_or_default();
        Err(AppError::Llm(format!("LLM returned {status}: {body_text}")))
    }
}

async fn test_oneshot(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    test_llm(State(state), auth_user, Json(body)).await
}

async fn test_whisper(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let whisper_type = body
        .get("whisper_type")
        .and_then(|v| v.as_str())
        .unwrap_or("remote");

    match whisper_type {
        "local" => {
            let has_local = cfg!(feature = "local-whisper");
            if has_local {
                Ok(Json(
                    json!({"status": "ok", "message": "Local whisper is available."}),
                ))
            } else {
                Err(AppError::BadRequest(
                    "Local whisper not compiled. Build with --features local-whisper.".into(),
                ))
            }
        }
        "groq" => {
            let api_key = body
                .get("groq_api_key")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if api_key.is_empty() {
                return Err(AppError::BadRequest("Groq API key is required.".into()));
            }
            Ok(Json(
                json!({"status": "ok", "message": "Groq API key provided."}),
            ))
        }
        _ => {
            let api_key = body
                .get("remote_api_key")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let base_url = body
                .get("remote_base_url")
                .and_then(|v| v.as_str())
                .unwrap_or("https://api.openai.com/v1");
            if api_key.is_empty() {
                return Err(AppError::BadRequest(
                    "Remote whisper API key required.".into(),
                ));
            }
            let client = reqwest::Client::new();
            let resp = client
                .get(&format!("{base_url}/models"))
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await
                .map_err(|e| AppError::Transcription(format!("Connection error: {e}")))?;

            if resp.status().is_success() {
                Ok(Json(
                    json!({"status": "ok", "message": "Remote whisper connection successful."}),
                ))
            } else {
                Err(AppError::Transcription(format!(
                    "Remote whisper returned {}",
                    resp.status()
                )))
            }
        }
    }
}

async fn whisper_capabilities() -> AppResult<Json<Value>> {
    Ok(Json(json!({
        "local_available": cfg!(feature = "local-whisper"),
        "remote_available": true,
        "groq_available": true,
    })))
}

async fn api_configured(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let llm = queries::get_llm_settings(&state.db).await?;
    let has_key = state
        .config
        .llm_api_key
        .as_ref()
        .or(state.config.openai_api_key.as_ref())
        .or(state.config.groq_api_key.as_ref())
        .or(llm.llm_api_key.as_ref())
        .map(|k| !k.is_empty())
        .unwrap_or(false);

    Ok(Json(json!({ "configured": has_key })))
}

async fn landing_status(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let app = queries::get_app_settings(&state.db).await?;
    let user_count = queries::count_users(&state.db).await?;
    let slots_remaining = app
        .user_limit_total
        .map(|limit| (limit - user_count).max(0));

    Ok(Json(json!({
        "enabled": app.enable_public_landing_page,
        "user_count": user_count,
        "user_limit": app.user_limit_total,
        "slots_remaining": slots_remaining,
        "require_auth": state.config.require_auth,
    })))
}
