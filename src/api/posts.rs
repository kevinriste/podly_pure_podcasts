use std::path::Path;

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::middleware::require_admin_user;
use crate::auth::AuthenticatedUser;
use crate::db::queries;
use crate::error::{AppError, AppResult};
use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/feeds/{feed_id}/posts", get(list_posts))
        .route("/api/posts/{p_guid}/stats", get(post_stats))
        .route("/api/posts/{p_guid}/status", get(post_status))
        .route("/api/posts/{p_guid}/whitelist", post(set_whitelist))
        .route("/api/posts/{p_guid}/process", post(process_post))
        .route("/api/posts/{p_guid}/reprocess", post(reprocess_post))
        .route("/api/posts/{p_guid}/audio", get(serve_audio))
        .route("/api/posts/{p_guid}/download", get(download_audio))
        .route(
            "/api/posts/{p_guid}/download/original",
            get(download_original),
        )
        .route("/api/posts/{p_guid}/processing-estimate", get(processing_estimate))
        .route("/api/posts/{p_guid}/json", get(post_json))
        .route("/post/{p_guid}/json", get(post_json))
        .route("/post/{p_guid}/debug", get(post_debug))
        // Legacy routes — both slash and dot variants
        .route("/post/{p_guid}/mp3", get(serve_audio_legacy))
        .route("/post/{p_guid}/original.mp3", get(download_original_legacy))
        // Dot-separated .mp3 route (used by RSS readers): /post/{guid}.mp3
        .route("/post/{p_guid_mp3}", get(serve_audio_dot_mp3))
}

#[derive(Deserialize)]
struct PostsQuery {
    page: Option<i64>,
    page_size: Option<i64>,
    /// Python accepts "1", "true", "yes", "on" — use custom deserializer
    #[serde(default, deserialize_with = "deserialize_truthy_bool")]
    whitelisted_only: bool,
}

fn deserialize_truthy_bool<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    let val: Option<String> = Option::deserialize(deserializer)?;
    match val.as_deref() {
        Some("1") | Some("true") | Some("yes") | Some("on") => Ok(true),
        _ => Ok(false),
    }
}

async fn list_posts(
    State(state): State<AppState>,
    AxumPath(feed_id): AxumPath<i64>,
    Query(q): Query<PostsQuery>,
) -> AppResult<Json<Value>> {
    let page = q.page.unwrap_or(1).max(1);
    let page_size = q.page_size.unwrap_or(25).clamp(1, 200);
    let whitelisted_only = q.whitelisted_only;

    let (posts, total) =
        queries::get_posts_by_feed(&state.db, feed_id, page, page_size, whitelisted_only).await?;

    let total_pages = (total + page_size - 1) / page_size;

    // Count whitelisted posts for this feed
    let whitelisted_total = queries::count_whitelisted_posts(&state.db, feed_id).await?;

    let items: Vec<Value> = posts
        .iter()
        .map(|p| {
            json!({
                "id": p.id, "guid": p.guid,
                "title": p.title, "description": p.description,
                "release_date": p.release_date, "duration": p.duration,
                "whitelisted": p.whitelisted, "image_url": p.image_url,
                "download_url": p.download_url,
                "download_count": p.download_count,
                "has_processed_audio": p.processed_audio_path.is_some(),
                "has_unprocessed_audio": p.unprocessed_audio_path.is_some(),
            })
        })
        .collect();

    Ok(Json(json!({
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "whitelisted_total": whitelisted_total,
    })))
}

async fn post_stats(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> AppResult<Json<Value>> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let segments = queries::get_segments_by_post(&state.db, post.id).await?;
    let identifications = queries::get_identifications_by_post(&state.db, post.id).await?;
    let model_calls = queries::get_model_calls_by_post(&state.db, post.id).await?;

    let total_segments = segments.len();
    let ad_idents: Vec<_> = identifications.iter().filter(|i| i.label == "ad").collect();
    let ad_count = ad_idents.len();
    let content_count = total_segments.saturating_sub(ad_count);

    // Calculate ad time
    let total_duration: f64 = segments.iter().map(|s| s.end_time - s.start_time).sum();
    let ad_duration: f64 = ad_idents
        .iter()
        .filter_map(|i| segments.iter().find(|s| s.id == i.transcript_segment_id))
        .map(|s| s.end_time - s.start_time)
        .sum();
    let ad_percentage = if total_duration > 0.0 {
        ((ad_duration / total_duration) * 1000.0).round() / 10.0
    } else {
        0.0
    };

    // Build ad blocks (merged time windows)
    let mut ad_windows: Vec<(f64, f64)> = ad_idents
        .iter()
        .filter_map(|i| segments.iter().find(|s| s.id == i.transcript_segment_id))
        .map(|s| (s.start_time, s.end_time))
        .collect();
    ad_windows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let ad_blocks = merge_time_windows(&ad_windows, 1.0);

    // Model call status counts
    let mut model_call_statuses = serde_json::Map::new();
    let mut model_types = serde_json::Map::new();
    for mc in &model_calls {
        let counter = model_call_statuses.entry(mc.status.clone()).or_insert(json!(0));
        *counter = json!(counter.as_i64().unwrap_or(0) + 1);
        let counter = model_types.entry(mc.model_name.clone()).or_insert(json!(0));
        *counter = json!(counter.as_i64().unwrap_or(0) + 1);
    }

    // Detect ad_detection_strategy
    let feed_strategy: Option<String> = sqlx::query_as::<_, (String,)>(
        "SELECT ad_detection_strategy FROM feed WHERE id = ?",
    )
    .bind(post.feed_id)
    .fetch_optional(&state.db)
    .await
    .ok()
    .flatten()
    .map(|(s,)| s);
    // Resolve "inherit" to the global default (like Python does)
    let ad_detection_strategy = match feed_strategy.as_deref() {
        Some("inherit") | None => {
            let app = queries::get_app_settings(&state.db).await?;
            app.ad_detection_strategy.clone()
        }
        Some(other) => other.to_string(),
    };

    // Build model_calls detail array
    let model_calls_json: Vec<Value> = model_calls
        .iter()
        .map(|mc| {
            json!({
                "id": mc.id,
                "model_name": mc.model_name,
                "status": mc.status,
                "segment_range": format!("{}-{}", mc.first_segment_sequence_num, mc.last_segment_sequence_num),
                "first_segment_sequence_num": mc.first_segment_sequence_num,
                "last_segment_sequence_num": mc.last_segment_sequence_num,
                "timestamp": mc.timestamp,
                "retry_attempts": mc.retry_attempts,
                "error_message": mc.error_message,
                "prompt": mc.prompt,
                "response": mc.response,
            })
        })
        .collect();

    // Build transcript segments with identifications
    let transcript_segments_json: Vec<Value> = segments
        .iter()
        .map(|s| {
            let seg_idents: Vec<Value> = identifications
                .iter()
                .filter(|i| i.transcript_segment_id == s.id)
                .map(|i| {
                    json!({
                        "id": i.id,
                        "label": i.label,
                        "confidence": i.confidence.map(|c| (c * 100.0).round() / 100.0),
                        "model_call_id": i.model_call_id,
                    })
                })
                .collect();

            let primary_label = seg_idents
                .first()
                .and_then(|i| i.get("label"))
                .and_then(|l| l.as_str())
                .unwrap_or("content");
            let mixed = seg_idents.len() > 1
                && seg_idents
                    .iter()
                    .any(|i| i.get("label").and_then(|l| l.as_str()) != Some(primary_label));

            json!({
                "id": s.id,
                "sequence_num": s.sequence_num,
                "start_time": (s.start_time * 10.0).round() / 10.0,
                "end_time": (s.end_time * 10.0).round() / 10.0,
                "text": s.text,
                "primary_label": primary_label,
                "mixed": mixed,
                "identifications": seg_idents,
            })
        })
        .collect();

    // Build identifications array
    let identifications_json: Vec<Value> = identifications
        .iter()
        .map(|i| {
            let seg = segments.iter().find(|s| s.id == i.transcript_segment_id);
            json!({
                "id": i.id,
                "transcript_segment_id": i.transcript_segment_id,
                "label": i.label,
                "confidence": i.confidence.map(|c| (c * 100.0).round() / 100.0),
                "model_call_id": i.model_call_id,
                "segment_sequence_num": seg.map(|s| s.sequence_num),
                "segment_start_time": seg.map(|s| (s.start_time * 10.0).round() / 10.0),
                "segment_end_time": seg.map(|s| (s.end_time * 10.0).round() / 10.0),
                "segment_text": seg.map(|s| &s.text),
            })
        })
        .collect();

    Ok(Json(json!({
        "post": {
            "guid": post.guid,
            "title": post.title,
            "duration": post.duration,
            "release_date": post.release_date,
            "whitelisted": post.whitelisted,
            "has_processed_audio": post.processed_audio_path.is_some(),
            "download_count": post.download_count,
        },
        "ad_detection_strategy": ad_detection_strategy,
        "processing_stats": {
            "total_segments": total_segments,
            "total_model_calls": model_calls.len(),
            "total_identifications": identifications.len(),
            "content_segments": content_count,
            "ad_segments_count": ad_count,
            "ad_percentage": ad_percentage,
            "estimated_ad_time_seconds": (ad_duration * 10.0).round() / 10.0,
            "ad_blocks": ad_blocks.iter().map(|(s, e)| json!({
                "start_time": (*s * 10.0).round() / 10.0,
                "end_time": (*e * 10.0).round() / 10.0,
            })).collect::<Vec<_>>(),
            "model_call_statuses": Value::Object(model_call_statuses),
            "model_types": Value::Object(model_types),
        },
        "model_calls": model_calls_json,
        "transcript_segments": transcript_segments_json,
        "identifications": identifications_json,
        "chapters": Value::Null,
    })))
}

fn merge_time_windows(windows: &[(f64, f64)], gap_seconds: f64) -> Vec<(f64, f64)> {
    if windows.is_empty() {
        return vec![];
    }
    let mut merged: Vec<(f64, f64)> = vec![windows[0]];
    for &(start, end) in &windows[1..] {
        let last = merged.last_mut().unwrap();
        if start <= last.1 + gap_seconds {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

async fn post_status(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    let result = state.jobs_manager.get_post_status(&p_guid).await;
    // Python returns 404 for NOT_FOUND, 400 for other error_codes, 200 otherwise
    let status_code = match result.get("error_code").and_then(|c| c.as_str()) {
        Some("NOT_FOUND") => StatusCode::NOT_FOUND,
        Some(_) => StatusCode::BAD_REQUEST,
        None => StatusCode::OK,
    };
    Ok((status_code, Json(result)).into_response())
}

#[derive(Deserialize)]
struct WhitelistRequest {
    whitelisted: Option<bool>,
    trigger_processing: Option<bool>,
}

async fn set_whitelist(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<WhitelistRequest>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let whitelisted = body.whitelisted.unwrap_or(true);
    queries::set_post_whitelist(&state.db, post.id, whitelisted).await?;

    let mut response = json!({
        "guid": p_guid,
        "whitelisted": whitelisted,
        "message": "Whitelist status updated successfully",
    });

    if whitelisted && body.trigger_processing.unwrap_or(false) {
        let user_id = get_auth_user_id(&auth_user);
        let job_result = state
            .jobs_manager
            .start_post_processing(&p_guid, user_id, user_id)
            .await;
        response["processing_job"] = job_result;
    }

    Ok(Json(response))
}

async fn process_post(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFoundMsg("Post not found.".into()))?;

    // Check feed exists (Python parity)
    let _feed = queries::get_feed_by_id(&state.db, post.feed_id)
        .await?
        .ok_or(AppError::NotFoundMsg("Feed not found".into()))?;

    // Check post is whitelisted (Python parity)
    if !post.whitelisted {
        return Err(AppError::BadRequest("Post is not whitelisted.".into()));
    }

    // Check if already processed (Python returns 200 with download URL)
    if let Some(ref audio_path) = post.processed_audio_path {
        if Path::new(audio_path).exists() {
            return Ok(Json(json!({
                "status": "already_processed",
                "message": "Post is already processed.",
                "download_url": format!("/api/posts/{}/download", p_guid),
            })));
        }
    }

    let user_id = get_auth_user_id(&auth_user);
    let result = state
        .jobs_manager
        .start_post_processing(&p_guid, user_id, user_id)
        .await;

    Ok(Json(result))
}

#[derive(Deserialize)]
struct ReprocessRequest {
    force_retranscribe: Option<bool>,
}

async fn reprocess_post(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<ReprocessRequest>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFoundMsg("Post not found.".into()))?;

    // Check feed exists (Python parity)
    let _feed = queries::get_feed_by_id(&state.db, post.feed_id)
        .await?
        .ok_or(AppError::NotFoundMsg("Feed not found".into()))?;

    // Check post is whitelisted (Python parity)
    if !post.whitelisted {
        return Err(AppError::BadRequest("Post is not whitelisted.".into()));
    }

    // Cancel existing jobs
    state.jobs_manager.cancel_post_jobs(&p_guid).await;

    // Clear processing data
    if body.force_retranscribe.unwrap_or(false) {
        // Full clear: transcript + identifications + model calls
        queries::clear_post_processing_data(&state.db, post.id).await?;
    } else {
        // Only clear identifications and model calls (keep transcript)
        queries::clear_post_identifications(&state.db, post.id).await?;
    }

    // Start reprocessing
    let user_id = get_auth_user_id(&auth_user);
    let result = state
        .jobs_manager
        .start_post_processing(&p_guid, user_id, user_id)
        .await;

    Ok(Json(result))
}

async fn serve_audio(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    if !post.whitelisted {
        return Ok(not_whitelisted_response());
    }

    match post.processed_audio_path.as_deref().filter(|p| Path::new(p).exists()) {
        Some(audio_path) => serve_file_with_range(audio_path, false, None, &headers).await,
        None => Ok(audio_not_ready_response()),
    }
}

async fn download_audio(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    if !post.whitelisted {
        // Check if autoprocess_on_download is enabled
        let app = queries::get_app_settings(&state.db).await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("{e}")))?;
        if !app.autoprocess_on_download {
            return Ok(not_whitelisted_response());
        }
        // Auto-whitelist the post
        let _ = queries::set_post_whitelist(&state.db, post.id, true).await;
        tracing::info!("Auto-whitelisted post {} on download request", p_guid);
    }

    let audio_path = post
        .processed_audio_path
        .as_deref()
        .filter(|p| Path::new(p).exists());

    match audio_path {
        Some(path) => {
            let _ = queries::increment_download_count(&state.db, post.id).await;
            let download_name = format!("{}.mp3", post.title);
            serve_file_with_range(path, true, Some(&download_name), &headers).await
        }
        None => {
            // Check autoprocess_on_download for auto-processing
            let app = queries::get_app_settings(&state.db).await
                .map_err(|e| AppError::Internal(anyhow::anyhow!("{e}")))?;
            if app.autoprocess_on_download {
                // Queue processing and return a 202
                let _ = state.jobs_manager.start_post_processing(&p_guid, None, None).await;
                return Ok((
                    StatusCode::ACCEPTED,
                    Json(json!({
                        "error": "Audio not ready",
                        "error_code": "AUDIO_NOT_READY",
                        "message": "Processing has been queued. Please try again later.",
                    })),
                ).into_response());
            }
            Ok(audio_not_ready_response())
        }
    }
}

async fn download_original(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    if !post.whitelisted {
        return Ok(not_whitelisted_response());
    }

    let audio_path = post
        .unprocessed_audio_path
        .as_deref()
        .filter(|p| Path::new(p).exists())
        .ok_or(AppError::NotFound)?;

    let _ = queries::increment_download_count(&state.db, post.id).await;
    let download_name = format!("{}_original.mp3", post.title);
    serve_file_with_range(audio_path, true, Some(&download_name), &headers).await
}

fn not_whitelisted_response() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({
            "error": "Post not whitelisted",
            "error_code": "NOT_WHITELISTED",
        })),
    )
        .into_response()
}

fn audio_not_ready_response() -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": "Audio not ready",
            "error_code": "AUDIO_NOT_READY",
            "message": "The processed audio is not yet available.",
        })),
    )
        .into_response()
}

async fn processing_estimate(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    // Python: returns 60.0 minutes when duration is None or 0
    let estimated_minutes = match post.duration {
        Some(d) if d > 0 => (d as f64 / 60.0).max(1.0),
        _ => 60.0,
    };

    Ok(Json(json!({
        "post_guid": p_guid,
        "estimated_minutes": estimated_minutes,
        "can_process": true,
        "reason": null,
    })))
}

async fn post_json(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> AppResult<Json<Value>> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let segment_count = queries::count_segments_by_post(&state.db, post.id).await?;
    let model_call_count = queries::count_model_calls_by_post(&state.db, post.id).await?;

    // Get transcript sample (first 5 segments, matching Python)
    let segments = queries::get_segments_by_post(&state.db, post.id).await?;
    let transcript_sample: Vec<Value> = segments
        .iter()
        .take(5)
        .map(|s| {
            let truncated_text = if s.text.len() > 100 {
                format!("{}...", &s.text[..100])
            } else {
                s.text.clone()
            };
            json!({
                "id": s.id,
                "sequence_num": s.sequence_num,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": truncated_text,
            })
        })
        .collect();

    // Get whisper model calls (matching Python)
    let whisper_model_calls: Vec<Value> = sqlx::query_as::<_, (i64, String, String, Option<i64>, Option<i64>, Option<String>, Option<String>, Option<String>)>(
        "SELECT id, model_name, status, first_segment_sequence_num, last_segment_sequence_num, timestamp, response, error_message FROM model_call WHERE post_id = ? AND model_name LIKE '%whisper%'",
    )
    .bind(post.id)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|mc| {
        let response_preview = mc.6.as_deref().map(|r| {
            if r.len() > 100 { format!("{}...", &r[..100]) } else { r.to_string() }
        });
        json!({
            "id": mc.0,
            "model_name": mc.1,
            "status": mc.2,
            "first_segment": mc.3,
            "last_segment": mc.4,
            "timestamp": mc.5,
            "response": response_preview,
            "error": mc.7,
        })
    })
    .collect();

    Ok(Json(json!({
        "id": post.id, "feed_id": post.feed_id, "guid": post.guid,
        "title": post.title,
        "unprocessed_audio_path": post.unprocessed_audio_path,
        "processed_audio_path": post.processed_audio_path,
        "has_processed_audio": post.processed_audio_path.is_some(),
        "has_unprocessed_audio": post.unprocessed_audio_path.is_some(),
        "transcript_segment_count": segment_count,
        "transcript_sample": transcript_sample,
        "model_call_count": model_call_count,
        "whisper_model_calls": whisper_model_calls,
        "whitelisted": post.whitelisted,
        "download_count": post.download_count,
    })))
}

async fn post_debug(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> AppResult<Json<Value>> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let segments = queries::get_segments_by_post(&state.db, post.id).await?;

    let model_calls: Vec<Value> = sqlx::query_as::<_, (i64, i64, Option<i64>, Option<i64>, String, Option<String>, Option<String>, String, Option<String>, Option<String>, i64)>(
        "SELECT id, post_id, first_segment_sequence_num, last_segment_sequence_num, model_name, prompt, response, status, error_message, timestamp, retry_attempts FROM model_call WHERE post_id = ? ORDER BY model_name, first_segment_sequence_num",
    )
    .bind(post.id)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|mc| json!({
        "id": mc.0, "post_id": mc.1,
        "first_segment_sequence_num": mc.2, "last_segment_sequence_num": mc.3,
        "model_name": mc.4, "prompt": mc.5, "response": mc.6,
        "status": mc.7, "error_message": mc.8, "timestamp": mc.9,
        "retry_attempts": mc.10,
    }))
    .collect();

    let identifications: Vec<Value> = sqlx::query_as::<_, (i64, i64, Option<i64>, f64, String)>(
        "SELECT i.id, i.transcript_segment_id, i.model_call_id, i.confidence, i.label FROM identification i JOIN transcript_segment ts ON i.transcript_segment_id = ts.id WHERE ts.post_id = ? ORDER BY ts.sequence_num",
    )
    .bind(post.id)
    .fetch_all(&state.db)
    .await
    .unwrap_or_default()
    .into_iter()
    .map(|id| json!({
        "id": id.0, "transcript_segment_id": id.1,
        "model_call_id": id.2, "confidence": id.3, "label": id.4,
    }))
    .collect();

    // Count content vs ad segments
    let mut content_count = 0i64;
    let mut ad_count = 0i64;
    // Group identifications by segment to determine primary label
    let mut seg_labels: std::collections::HashMap<i64, &str> = std::collections::HashMap::new();
    for id_val in &identifications {
        let seg_id = id_val["transcript_segment_id"].as_i64().unwrap_or(0);
        let label = id_val["label"].as_str().unwrap_or("content");
        let conf = id_val["confidence"].as_f64().unwrap_or(0.0);
        let entry = seg_labels.entry(seg_id).or_insert(label);
        // Higher confidence wins
        if label == "ad" && conf > 0.5 {
            *entry = "ad";
        }
    }
    for seg in &segments {
        match seg_labels.get(&seg.id).copied() {
            Some("ad") => ad_count += 1,
            _ => content_count += 1,
        }
    }

    let stats = json!({
        "total_segments": segments.len(),
        "total_model_calls": model_calls.len(),
        "total_identifications": identifications.len(),
        "content_segments": content_count,
        "ad_segments_count": ad_count,
        "download_count": post.download_count,
    });

    Ok(Json(json!({
        "post": {
            "id": post.id, "guid": post.guid, "title": post.title,
            "feed_id": post.feed_id, "whitelisted": post.whitelisted,
        },
        "stats": stats,
        "model_calls": model_calls,
        "transcript_segments": segments.iter().map(|s| json!({
            "id": s.id, "sequence_num": s.sequence_num,
            "start_time": s.start_time, "end_time": s.end_time,
            "text": s.text,
        })).collect::<Vec<_>>(),
        "identifications": identifications,
    })))
}

// Legacy routes — Python routes /post/{guid}.mp3 to download (Content-Disposition: attachment)
async fn serve_audio_legacy(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    download_audio(State(state), headers, AxumPath(p_guid)).await
}

async fn serve_audio_dot_mp3(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(p_guid_mp3): AxumPath<String>,
) -> Result<Response, AppError> {
    // Handle /post/{guid}.mp3 — strip the .mp3 suffix
    let p_guid = match p_guid_mp3.strip_suffix(".mp3") {
        Some(guid) => guid.to_string(),
        None => return Err(AppError::NotFound),
    };
    download_audio(State(state), headers, AxumPath(p_guid)).await
}

async fn download_original_legacy(
    State(state): State<AppState>,
    headers: HeaderMap,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    download_original(State(state), headers, AxumPath(p_guid)).await
}

async fn serve_file_with_range(
    path: &str,
    as_attachment: bool,
    download_name: Option<&str>,
    headers: &HeaderMap,
) -> Result<Response, AppError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|_| AppError::NotFound)?;

    let total_len = bytes.len();

    let filename = download_name
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            Path::new(path)
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("audio.mp3")
                .to_string()
        });

    let content_type = if path.ends_with(".mp3") {
        "audio/mpeg"
    } else if path.ends_with(".ogg") {
        "audio/ogg"
    } else {
        "application/octet-stream"
    };

    // Parse Range header for byte-range requests (podcast player seeking)
    if let Some(range_val) = headers.get(axum::http::header::RANGE).and_then(|v| v.to_str().ok()) {
        if let Some(range) = parse_byte_range(range_val, total_len) {
            let (start, end) = range;
            let slice = bytes[start..=end].to_vec();
            let content_len = slice.len();

            let mut builder = axum::http::Response::builder()
                .status(StatusCode::PARTIAL_CONTENT)
                .header(axum::http::header::CONTENT_TYPE, content_type)
                .header(axum::http::header::ACCEPT_RANGES, "bytes")
                .header(axum::http::header::CONTENT_LENGTH, content_len.to_string())
                .header(
                    axum::http::header::CONTENT_RANGE,
                    format!("bytes {start}-{end}/{total_len}"),
                );

            if as_attachment {
                builder = builder.header(
                    axum::http::header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{filename}\""),
                );
            }

            return Ok(builder
                .body(axum::body::Body::from(slice))
                .unwrap()
                .into_response());
        }
    }

    // Full response (no Range header or unparseable range)
    let mut builder = axum::http::Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .header(axum::http::header::ACCEPT_RANGES, "bytes")
        .header(axum::http::header::CONTENT_LENGTH, total_len.to_string());

    if as_attachment {
        builder = builder.header(
            axum::http::header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        );
    }

    Ok(builder
        .body(axum::body::Body::from(bytes))
        .unwrap()
        .into_response())
}

/// Parse "bytes=START-END" range header. Returns (start, end) inclusive.
fn parse_byte_range(range_str: &str, total: usize) -> Option<(usize, usize)> {
    let range_str = range_str.strip_prefix("bytes=")?;
    let mut parts = range_str.splitn(2, '-');
    let start_str = parts.next()?.trim();
    let end_str = parts.next()?.trim();

    if start_str.is_empty() {
        // Suffix range: bytes=-500 means last 500 bytes
        let suffix_len: usize = end_str.parse().ok()?;
        let start = total.saturating_sub(suffix_len);
        Some((start, total - 1))
    } else {
        let start: usize = start_str.parse().ok()?;
        if start >= total {
            return None;
        }
        let end = if end_str.is_empty() {
            total - 1
        } else {
            end_str.parse::<usize>().ok()?.min(total - 1)
        };
        if start > end {
            return None;
        }
        Some((start, end))
    }
}

fn get_auth_user_id(auth_user: &Option<Extension<AuthenticatedUser>>) -> Option<i64> {
    auth_user.as_ref().map(|Extension(u)| u.id)
}
