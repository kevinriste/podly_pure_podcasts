use axum::extract::State;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde_json::{json, Value};

use crate::auth::middleware::require_admin_user;
use crate::auth::AuthenticatedUser;
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

    // Mask API keys
    fn mask(val: &Option<String>) -> Option<String> {
        val.as_ref().and_then(|s| {
            let s = s.trim();
            if s.is_empty() { return None; }
            if s.len() <= 8 { return Some("****".into()); }
            Some(format!("{}****{}", &s[..4], &s[s.len() - 4..]))
        })
    }

    // Apply env overrides
    let effective_api_key = state.config.llm_api_key.clone().or(llm.llm_api_key.clone());
    let effective_whisper_type = state.config.whisper_type.clone().unwrap_or(whisper.whisper_type.clone());

    Ok(Json(json!({
        "llm": {
            "llm_api_key": mask(&effective_api_key),
            "llm_model": state.config.llm_model.as_deref().unwrap_or(&llm.llm_model),
            "oneshot_model": state.config.oneshot_model.as_deref().or(llm.oneshot_model.as_deref()),
            "openai_base_url": state.config.openai_base_url.as_deref().or(llm.openai_base_url.as_deref()),
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
            "whisper_type": effective_whisper_type,
            "local_model": whisper.local_model,
            "remote_model": whisper.remote_model,
            "remote_api_key": mask(&whisper.remote_api_key),
            "remote_base_url": whisper.remote_base_url,
            "remote_language": whisper.remote_language,
            "remote_timeout_sec": whisper.remote_timeout_sec,
            "remote_chunksize_mb": whisper.remote_chunksize_mb,
            "groq_api_key": mask(&state.config.groq_api_key.clone().or(whisper.groq_api_key.clone())),
            "groq_model": whisper.groq_model,
            "groq_language": whisper.groq_language,
            "groq_max_retries": whisper.groq_max_retries,
        },
        "processing": {
            "system_prompt_path": processing.system_prompt_path,
            "user_prompt_template_path": processing.user_prompt_template_path,
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
            "ad_detection_strategy": app.ad_detection_strategy,
            "enable_public_landing_page": app.enable_public_landing_page,
            "user_limit_total": app.user_limit_total,
            "autoprocess_on_download": app.autoprocess_on_download,
        },
        "chapter_filter": {
            "default_filter_strings": chapter.default_filter_strings,
        },
    })))
}

async fn update_config(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let now = chrono::Utc::now().to_rfc3339();

    // Update LLM settings
    if let Some(llm) = body.get("llm") {
        let api_key = llm.get("llm_api_key").and_then(|v| v.as_str()).map(|s| s.to_string());
        let model = llm.get("llm_model").and_then(|v| v.as_str()).map(|s| s.to_string());

        if let Some(key) = &api_key {
            if !key.contains("****") {
                sqlx::query("UPDATE llm_settings SET llm_api_key = ?, updated_at = ? WHERE id = 1")
                    .bind(key)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
            }
        }
        if let Some(model) = &model {
            sqlx::query("UPDATE llm_settings SET llm_model = ?, updated_at = ? WHERE id = 1")
                .bind(model)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("openai_base_url").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE llm_settings SET openai_base_url = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("openai_timeout").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE llm_settings SET openai_timeout = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("openai_max_tokens").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE llm_settings SET openai_max_tokens = ?, updated_at = ? WHERE id = 1")
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
        if let Some(v) = llm.get("enable_boundary_refinement").and_then(|v| v.as_bool()) {
            sqlx::query("UPDATE llm_settings SET enable_boundary_refinement = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("oneshot_model").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE llm_settings SET oneshot_model = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
    }

    // Update whisper settings
    if let Some(w) = body.get("whisper") {
        if let Some(v) = w.get("whisper_type").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE whisper_settings SET whisper_type = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = w.get("remote_model").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE whisper_settings SET remote_model = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = w.get("remote_api_key").and_then(|v| v.as_str()) {
            if !v.contains("****") {
                sqlx::query("UPDATE whisper_settings SET remote_api_key = ?, updated_at = ? WHERE id = 1")
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
            }
        }
        if let Some(v) = w.get("groq_api_key").and_then(|v| v.as_str()) {
            if !v.contains("****") {
                sqlx::query("UPDATE whisper_settings SET groq_api_key = ?, updated_at = ? WHERE id = 1")
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
            }
        }
        if let Some(v) = w.get("groq_model").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE whisper_settings SET groq_model = ?, updated_at = ? WHERE id = 1")
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
        if let Some(v) = o.get("min_ad_segment_separation_seconds").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE output_settings SET min_ad_segement_separation_seconds = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = o.get("min_confidence").and_then(|v| v.as_f64()) {
            sqlx::query("UPDATE output_settings SET min_confidence = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
    }

    // Update app settings
    if let Some(a) = body.get("app") {
        if let Some(v) = a.get("automatically_whitelist_new_episodes").and_then(|v| v.as_bool()) {
            sqlx::query("UPDATE app_settings SET automatically_whitelist_new_episodes = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a.get("background_update_interval_minute").and_then(|v| v.as_i64()) {
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
        if let Some(v) = a.get("ad_detection_strategy").and_then(|v| v.as_str()) {
            sqlx::query("UPDATE app_settings SET ad_detection_strategy = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a.get("enable_public_landing_page").and_then(|v| v.as_bool()) {
            sqlx::query("UPDATE app_settings SET enable_public_landing_page = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = a.get("autoprocess_on_download").and_then(|v| v.as_bool()) {
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

    let api_key = body.get("api_key").and_then(|v| v.as_str()).unwrap_or("");
    let model = body.get("model").and_then(|v| v.as_str()).unwrap_or("gpt-4");
    let base_url = body.get("base_url").and_then(|v| v.as_str()).unwrap_or("https://api.openai.com/v1");

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
        Ok(Json(json!({"status": "ok", "message": "LLM connection successful."})))
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
    // Same as test_llm but with oneshot model
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
                Ok(Json(json!({"status": "ok", "message": "Local whisper is available."})))
            } else {
                Err(AppError::BadRequest("Local whisper not compiled. Build with --features local-whisper.".into()))
            }
        }
        "groq" => {
            let api_key = body.get("groq_api_key").and_then(|v| v.as_str()).unwrap_or("");
            if api_key.is_empty() {
                return Err(AppError::BadRequest("Groq API key is required.".into()));
            }
            Ok(Json(json!({"status": "ok", "message": "Groq API key provided."})))
        }
        _ => {
            // Remote whisper
            let api_key = body.get("remote_api_key").and_then(|v| v.as_str()).unwrap_or("");
            let base_url = body.get("remote_base_url").and_then(|v| v.as_str()).unwrap_or("https://api.openai.com/v1");
            if api_key.is_empty() {
                return Err(AppError::BadRequest("Remote whisper API key required.".into()));
            }
            // Test connectivity
            let client = reqwest::Client::new();
            let resp = client
                .get(&format!("{base_url}/models"))
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await
                .map_err(|e| AppError::Transcription(format!("Connection error: {e}")))?;

            if resp.status().is_success() {
                Ok(Json(json!({"status": "ok", "message": "Remote whisper connection successful."})))
            } else {
                Err(AppError::Transcription(format!("Remote whisper returned {}", resp.status())))
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
        .or(llm.llm_api_key.as_ref())
        .map(|k| !k.is_empty())
        .unwrap_or(false);

    Ok(Json(json!({ "configured": has_key })))
}

async fn landing_status(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let app = queries::get_app_settings(&state.db).await?;
    let user_count = queries::count_users(&state.db).await?;
    Ok(Json(json!({
        "enabled": app.enable_public_landing_page,
        "user_count": user_count,
        "user_limit": app.user_limit_total,
        "require_auth": state.config.require_auth,
    })))
}
