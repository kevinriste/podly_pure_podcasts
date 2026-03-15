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
        Some(format!("{}...{}", &s[..4], &s[s.len() - 4..]))
    })
}

/// Build the `env_overrides` metadata map following PR #196's 12-factor model.
/// Each entry has `env_var`, `read_only: true`, and optionally masked value info.
fn build_env_override_metadata(config: &AppConfig, data: &Value) -> Value {
    let mut overrides = serde_json::Map::new();

    // Helper to register an override (matches Python shape — no read_only key)
    let mut register = |path: &str, env_var: &str, value: &Option<String>, secret: bool| {
        if let Some(val) = value {
            if val.is_empty() {
                return;
            }
            let mut entry = serde_json::Map::new();
            entry.insert("env_var".into(), json!(env_var));
            if secret {
                entry.insert("is_secret".into(), json!(true));
                entry.insert("value_preview".into(), json!(mask(value)));
            } else {
                entry.insert("value".into(), json!(val));
            }
            overrides.insert(path.into(), Value::Object(entry));
        }
    };

    // LLM overrides — report the actual env var name that was found (Python parity)
    let (llm_api_key, llm_api_key_env) = if config.llm_api_key.is_some() {
        (config.llm_api_key.clone(), "LLM_API_KEY")
    } else if config.openai_api_key.is_some() {
        (config.openai_api_key.clone(), "OPENAI_API_KEY")
    } else if config.groq_api_key.is_some() {
        (config.groq_api_key.clone(), "GROQ_API_KEY")
    } else {
        (None, "LLM_API_KEY")
    };
    register("llm.llm_api_key", llm_api_key_env, &llm_api_key, true);
    register(
        "llm.openai_base_url",
        "OPENAI_BASE_URL",
        &config.openai_base_url,
        false,
    );
    register("llm.llm_model", "LLM_MODEL", &config.llm_model, false);
    register("llm.oneshot_model", "ONESHOT_MODEL", &config.oneshot_model, false);

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
            register("groq.api_key", "GROQ_API_KEY", &config.groq_api_key, true);
            register("whisper.api_key", "GROQ_API_KEY", &config.groq_api_key, true);
            register(
                "whisper.model",
                "GROQ_WHISPER_MODEL",
                &config.groq_whisper_model,
                false,
            );
            register(
                "whisper.max_retries",
                "GROQ_MAX_RETRIES",
                &config.groq_max_retries.clone(),
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
    let _chapter = queries::get_chapter_filter_settings(&state.db).await?;

    // Build whisper payload — flatten to type-specific fields (matches Python)
    let effective_whisper_type = state
        .config
        .whisper_type
        .as_deref()
        .unwrap_or(&whisper.whisper_type);
    let mut whisper_payload = json!({ "whisper_type": effective_whisper_type });
    match effective_whisper_type {
        "local" => {
            whisper_payload
                .as_object_mut()
                .unwrap()
                .insert("model".into(), json!(&whisper.local_model));
        }
        "remote" => {
            let w = whisper_payload.as_object_mut().unwrap();
            w.insert("model".into(), json!(&whisper.remote_model));
            // Mask the api_key (Python pops api_key, inserts api_key_preview)
            let effective_key = state
                .config
                .whisper_remote_api_key
                .as_ref()
                .or(state.config.openai_api_key.as_ref())
                .cloned()
                .or(whisper.remote_api_key.clone());
            w.insert("api_key_preview".into(), json!(mask(&effective_key)));
            let effective_base = state
                .config
                .whisper_remote_base_url
                .as_deref()
                .unwrap_or(&whisper.remote_base_url);
            w.insert("base_url".into(), json!(effective_base));
            w.insert("language".into(), json!(&whisper.remote_language));
            w.insert("timeout_sec".into(), json!(whisper.remote_timeout_sec));
            w.insert("chunksize_mb".into(), json!(whisper.remote_chunksize_mb));
        }
        "groq" => {
            let w = whisper_payload.as_object_mut().unwrap();
            let effective_model = state
                .config
                .groq_whisper_model
                .as_deref()
                .unwrap_or(&whisper.groq_model);
            w.insert("model".into(), json!(effective_model));
            let effective_key = state
                .config
                .groq_api_key
                .clone()
                .or(whisper.groq_api_key.clone());
            w.insert("api_key_preview".into(), json!(mask(&effective_key)));
            w.insert("language".into(), json!(&whisper.groq_language));
            w.insert("max_retries".into(), json!(whisper.groq_max_retries));
        }
        _ => {}
    }

    // Build LLM section — mask api key like Python (pop llm_api_key, insert preview)
    let effective_llm_key = state
        .config
        .llm_api_key
        .as_ref()
        .or(state.config.openai_api_key.as_ref())
        .or(state.config.groq_api_key.as_ref())
        .cloned()
        .or(llm.llm_api_key.clone());
    let effective_model = state
        .config
        .llm_model
        .as_deref()
        .unwrap_or(&llm.llm_model);
    let effective_base_url = state
        .config
        .openai_base_url
        .clone()
        .or(llm.openai_base_url.clone());

    let data = json!({
        "llm": {
            "llm_api_key_preview": mask(&effective_llm_key),
            "llm_model": effective_model,
            "oneshot_model": state.config.oneshot_model.as_deref().or(llm.oneshot_model.as_deref()),
            "openai_base_url": effective_base_url,
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
        "whisper": whisper_payload,
        "processing": {
            "num_segments_to_input_to_prompt": processing.num_segments_to_input_to_prompt,
            "max_overlap_segments": processing.max_overlap_segments,
        },
        "output": {
            "fade_ms": output.fade_ms,
            "min_ad_segement_separation_seconds": output.min_ad_segment_separation_seconds,
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
    });

    // Build env_overrides metadata (tells frontend which fields are read-only)
    let env_overrides = build_env_override_metadata(&state.config, &data);

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
        if let Some(v) = llm.get("llm_max_input_tokens_per_call").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE llm_settings SET llm_max_input_tokens_per_call = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("llm_enable_token_rate_limiting").and_then(|v| v.as_bool()) {
            sqlx::query("UPDATE llm_settings SET llm_enable_token_rate_limiting = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("llm_max_input_tokens_per_minute").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE llm_settings SET llm_max_input_tokens_per_minute = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("enable_word_level_boundary_refinder").and_then(|v| v.as_bool()) {
            sqlx::query("UPDATE llm_settings SET enable_word_level_boundary_refinder = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("oneshot_max_chunk_duration_seconds").and_then(|v| v.as_f64()) {
            sqlx::query("UPDATE llm_settings SET oneshot_max_chunk_duration_seconds = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = llm.get("oneshot_chunk_overlap_seconds").and_then(|v| v.as_f64()) {
            sqlx::query("UPDATE llm_settings SET oneshot_chunk_overlap_seconds = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
    }

    // Update whisper settings — frontend sends flat/generic keys (model, api_key, etc.)
    // mapped to provider-specific DB columns based on whisper_type (matches Python)
    if let Some(w) = body.get("whisper") {
        // Determine effective whisper type (from payload or current DB value)
        let current_whisper = queries::get_whisper_settings(&state.db).await?;
        let wtype = w
            .get("whisper_type")
            .and_then(|v| v.as_str())
            .unwrap_or(&current_whisper.whisper_type);

        if w.get("whisper_type").is_some() {
            sqlx::query(
                "UPDATE whisper_settings SET whisper_type = ?, updated_at = ? WHERE id = 1",
            )
            .bind(wtype)
            .bind(&now)
            .execute(&state.db)
            .await?;
        }

        match wtype {
            "local" => {
                if let Some(v) = w.get("model").and_then(|v| v.as_str()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET local_model = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
            }
            "remote" => {
                if let Some(v) = w.get("model").and_then(|v| v.as_str()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET remote_model = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
                if let Some(v) = w.get("api_key").and_then(|v| v.as_str()) {
                    if !v.contains("****") && !v.contains("...") && !v.is_empty() {
                        sqlx::query("UPDATE whisper_settings SET remote_api_key = ?, updated_at = ? WHERE id = 1")
                            .bind(v)
                            .bind(&now)
                            .execute(&state.db)
                            .await?;
                    }
                }
                if let Some(v) = w.get("base_url").and_then(|v| v.as_str()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET remote_base_url = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
                if let Some(v) = w.get("language").and_then(|v| v.as_str()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET remote_language = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
                if let Some(v) = w.get("timeout_sec").and_then(|v| v.as_i64()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET remote_timeout_sec = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
                if let Some(v) = w.get("chunksize_mb").and_then(|v| v.as_i64()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET remote_chunksize_mb = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
            }
            "groq" => {
                if let Some(v) = w.get("api_key").and_then(|v| v.as_str()) {
                    if !v.contains("****") && !v.contains("...") && !v.is_empty() {
                        sqlx::query("UPDATE whisper_settings SET groq_api_key = ?, updated_at = ? WHERE id = 1")
                            .bind(v)
                            .bind(&now)
                            .execute(&state.db)
                            .await?;
                    }
                }
                if let Some(v) = w.get("model").and_then(|v| v.as_str()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET groq_model = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
                if let Some(v) = w.get("language").and_then(|v| v.as_str()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET groq_language = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
                if let Some(v) = w.get("max_retries").and_then(|v| v.as_i64()) {
                    sqlx::query(
                        "UPDATE whisper_settings SET groq_max_retries = ?, updated_at = ? WHERE id = 1",
                    )
                    .bind(v)
                    .bind(&now)
                    .execute(&state.db)
                    .await?;
                }
            }
            _ => {}
        }
    }

    // Update processing settings
    if let Some(p) = body.get("processing") {
        if let Some(v) = p
            .get("num_segments_to_input_to_prompt")
            .and_then(|v| v.as_i64())
        {
            sqlx::query("UPDATE processing_settings SET num_segments_to_input_to_prompt = ?, updated_at = ? WHERE id = 1")
                .bind(v)
                .bind(&now)
                .execute(&state.db)
                .await?;
        }
        if let Some(v) = p.get("max_overlap_segments").and_then(|v| v.as_i64()) {
            sqlx::query("UPDATE processing_settings SET max_overlap_segments = ?, updated_at = ? WHERE id = 1")
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
            .get("min_ad_segement_separation_seconds")
            .or_else(|| o.get("min_ad_segment_separation_seconds"))
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

    // Return updated config (sanitized, matches Python's PUT response)
    let updated = get_config(State(state), auth_user).await?;
    // Python returns just the config sections (no env_overrides wrapping)
    let full: Value = updated.0;
    if let Some(config) = full.get("config") {
        Ok(Json(config.clone()))
    } else {
        Ok(Json(full))
    }
}

async fn test_llm(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    // Python reads from body.llm.* and falls back to runtime config
    let llm_section = body.get("llm").unwrap_or(&body);
    let llm_db = queries::get_llm_settings(&state.db).await?;

    let api_key = llm_section
        .get("llm_api_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| state.config.llm_api_key.clone())
        .or_else(|| state.config.openai_api_key.clone())
        .or_else(|| state.config.groq_api_key.clone())
        .or_else(|| llm_db.llm_api_key.clone())
        .unwrap_or_default();
    let model = llm_section
        .get("llm_model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| state.config.llm_model.clone())
        .unwrap_or_else(|| llm_db.llm_model.clone());
    let base_url = llm_section
        .get("openai_base_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| state.config.openai_base_url.clone())
        .or_else(|| llm_db.openai_base_url.clone());
    let timeout: u64 = llm_section
        .get("openai_timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(llm_db.openai_timeout as u64);

    run_llm_probe(&api_key, &model, base_url.as_deref(), timeout, "LLM connection OK").await
}

async fn run_llm_probe(
    api_key: &str,
    model: &str,
    base_url: Option<&str>,
    _timeout: u64,
    success_message: &str,
) -> AppResult<Json<Value>> {
    if api_key.is_empty() {
        // Python returns 400 with {"ok": false, "error": "Missing llm_api_key"}
        return Ok(Json(json!({"ok": false, "error": "Missing llm_api_key"})));
    }

    // Use genai crate — handles OpenAI, Gemini, Anthropic, Groq, etc.
    let client = crate::llm::build_genai_client(api_key, model, base_url)
        .map_err(|e| AppError::Llm(format!("Client error: {e}")))?;

    let chat_req = genai::chat::ChatRequest::new(vec![
        genai::chat::ChatMessage::system("You are a healthcheck probe."),
        genai::chat::ChatMessage::user("ping"),
    ]);

    let options = genai::chat::ChatOptions::default()
        .with_max_tokens(10u32);

    let genai_model = crate::llm::to_genai_model(model);
    match client.exec_chat(&genai_model, chat_req, Some(&options)).await {
        Ok(_) => Ok(Json(json!({
            "ok": true,
            "message": success_message,
            "model": model,
            "base_url": base_url,
        }))),
        // Python returns 400 with {"ok": false, "error": "..."}
        Err(e) => Ok(Json(json!({"ok": false, "error": format!("LLM error: {e}")}))),
    }
}

async fn test_oneshot(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let llm_section = body.get("llm").unwrap_or(&body);
    let llm_db = queries::get_llm_settings(&state.db).await?;

    // Python uses get_effective_oneshot_api_key: ONESHOT_API_KEY -> LLM_API_KEY -> DB llm_api_key
    let api_key = llm_section
        .get("llm_api_key")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| state.config.llm_api_key.clone())
        .or_else(|| state.config.openai_api_key.clone())
        .or_else(|| state.config.groq_api_key.clone())
        .or_else(|| llm_db.llm_api_key.clone())
        .unwrap_or_default();

    // Read oneshot_model from the payload
    let model = llm_section
        .get("oneshot_model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| state.config.oneshot_model.clone())
        .or_else(|| llm_db.oneshot_model.clone())
        .unwrap_or_default();

    if model.is_empty() {
        return Ok(Json(json!({"ok": false, "error": "Missing oneshot_model. Configure One-shot Model first."})));
    }

    let base_url = llm_section
        .get("openai_base_url")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| state.config.openai_base_url.clone())
        .or_else(|| llm_db.openai_base_url.clone());
    let timeout: u64 = llm_section
        .get("openai_timeout")
        .and_then(|v| v.as_u64())
        .unwrap_or(llm_db.openai_timeout as u64);

    run_llm_probe(&api_key, &model, base_url.as_deref(), timeout, "One-shot connection OK").await
}

async fn test_whisper(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<Value>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    // Python reads from body.whisper.* and falls back to runtime config
    let whisper_section = body.get("whisper").unwrap_or(&body);
    let whisper_db = queries::get_whisper_settings(&state.db).await?;

    let effective_type = state
        .config
        .whisper_type
        .as_deref()
        .unwrap_or(&whisper_db.whisper_type);
    let whisper_type = whisper_section
        .get("whisper_type")
        .and_then(|v| v.as_str())
        .unwrap_or(effective_type);

    match whisper_type {
        "local" => {
            let has_local = cfg!(feature = "local-whisper");
            if has_local {
                Ok(Json(json!({"ok": true, "message": "Local whisper is available."})))
            } else {
                Ok(Json(json!({"ok": false, "error": "Local whisper not compiled. Build with --features local-whisper."})))
            }
        }
        "groq" => {
            let api_key = whisper_section
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| state.config.groq_api_key.clone())
                .or_else(|| whisper_db.groq_api_key.clone())
                .unwrap_or_default();
            if api_key.is_empty() {
                return Ok(Json(json!({"ok": false, "error": "Groq API key is required."})));
            }
            // Actually test the Groq connection (Python calls groq.models.list())
            let client = reqwest::Client::new();
            match client
                .get("https://api.groq.com/openai/v1/models")
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    Ok(Json(json!({"ok": true, "message": "Groq whisper connection OK"})))
                }
                Ok(resp) => {
                    Ok(Json(json!({"ok": false, "error": format!("Groq returned {}", resp.status())})))
                }
                Err(e) => {
                    Ok(Json(json!({"ok": false, "error": format!("Groq connection error: {e}")})))
                }
            }
        }
        _ => {
            let api_key = whisper_section
                .get("api_key")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| state.config.whisper_remote_api_key.clone())
                .or_else(|| state.config.openai_api_key.clone())
                .or_else(|| whisper_db.remote_api_key.clone())
                .unwrap_or_default();
            let base_url = whisper_section
                .get("base_url")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .or_else(|| state.config.whisper_remote_base_url.clone())
                .or_else(|| state.config.openai_base_url.clone())
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            if api_key.is_empty() {
                return Ok(Json(json!({"ok": false, "error": "Remote whisper API key required."})));
            }
            let client = reqwest::Client::new();
            match client
                .get(&format!("{base_url}/models"))
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    Ok(Json(json!({"ok": true, "message": "Remote whisper connection OK", "base_url": base_url})))
                }
                Ok(resp) => {
                    Ok(Json(json!({"ok": false, "error": format!("Remote whisper returned {}", resp.status())})))
                }
                Err(e) => {
                    Ok(Json(json!({"ok": false, "error": format!("Connection error: {e}")})))
                }
            }
        }
    }
}

async fn whisper_capabilities(
    auth_user: Option<Extension<AuthenticatedUser>>,
    State(state): State<AppState>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;
    Ok(Json(json!({
        "local_available": cfg!(feature = "local-whisper"),
    })))
}

async fn api_configured(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Json<Value> {
    // Python requires admin but catches all errors defensively
    if require_admin_user(&auth_user, state.config.require_auth).is_err() {
        return Json(json!({ "configured": false }));
    }
    // Python catches all exceptions and returns {configured: false}
    let has_key = match queries::get_llm_settings(&state.db).await {
        Ok(llm) => state
            .config
            .llm_api_key
            .as_ref()
            .or(state.config.openai_api_key.as_ref())
            .or(state.config.groq_api_key.as_ref())
            .or(llm.llm_api_key.as_ref())
            .map(|k| !k.is_empty())
            .unwrap_or(false),
        Err(_) => false,
    };

    Json(json!({ "configured": has_key }))
}

async fn landing_status(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let app = queries::get_app_settings(&state.db).await?;
    let user_count = queries::count_users(&state.db).await?;
    let slots_remaining = app
        .user_limit_total
        .map(|limit| (limit - user_count).max(0));

    Ok(Json(json!({
        "landing_page_enabled": app.enable_public_landing_page,
        "user_count": user_count,
        "user_limit_total": app.user_limit_total,
        "slots_remaining": slots_remaining,
        "require_auth": state.config.require_auth,
    })))
}
