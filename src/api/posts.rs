use std::path::Path;

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::StatusCode;
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
        .route("/api/posts/{p_guid}/estimate", get(processing_estimate))
        .route("/api/posts/{p_guid}/json", get(post_json))
        // Legacy routes
        .route("/post/{p_guid}/mp3", get(serve_audio_legacy))
        .route("/post/{p_guid}/original.mp3", get(download_original_legacy))
}

#[derive(Deserialize)]
struct PostsQuery {
    page: Option<i64>,
    page_size: Option<i64>,
    whitelisted_only: Option<bool>,
}

async fn list_posts(
    State(state): State<AppState>,
    AxumPath(feed_id): AxumPath<i64>,
    Query(q): Query<PostsQuery>,
) -> AppResult<Json<Value>> {
    let page = q.page.unwrap_or(1).max(1);
    let page_size = q.page_size.unwrap_or(50).clamp(1, 200);
    let whitelisted_only = q.whitelisted_only.unwrap_or(false);

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
    let ad_detection_strategy = feed_strategy.as_deref().unwrap_or("llm");

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
) -> AppResult<Json<Value>> {
    let result = state.jobs_manager.get_post_status(&p_guid).await;
    Ok(Json(result))
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

    if whitelisted && body.trigger_processing.unwrap_or(false) {
        let user_id = get_auth_user_id(&auth_user);
        state
            .jobs_manager
            .start_post_processing(&p_guid, user_id, user_id)
            .await;
    }

    Ok(Json(json!({"status": "ok", "whitelisted": whitelisted})))
}

async fn process_post(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let _post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let user_id = get_auth_user_id(&auth_user);
    let result = state
        .jobs_manager
        .start_post_processing(&p_guid, user_id, user_id)
        .await;

    Ok(Json(result))
}

#[derive(Deserialize)]
struct ReprocessRequest {
    clear_mode: Option<String>,
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
        .ok_or(AppError::NotFound)?;

    // Cancel existing jobs
    state.jobs_manager.cancel_post_jobs(&p_guid).await;

    // Clear processing data
    let mode = body.clear_mode.as_deref().unwrap_or("full");
    match mode {
        "identifications" => {
            queries::clear_post_identifications(&state.db, post.id).await?;
        }
        _ => {
            queries::clear_post_processing_data(&state.db, post.id).await?;
        }
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
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let audio_path = post
        .processed_audio_path
        .as_deref()
        .filter(|p| Path::new(p).exists())
        .ok_or(AppError::NotFound)?;

    serve_file_response(audio_path, false).await
}

async fn download_audio(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let audio_path = post
        .processed_audio_path
        .as_deref()
        .filter(|p| Path::new(p).exists())
        .ok_or(AppError::NotFound)?;

    let _ = queries::increment_download_count(&state.db, post.id).await;

    serve_file_response(audio_path, true).await
}

async fn download_original(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let audio_path = post
        .unprocessed_audio_path
        .as_deref()
        .filter(|p| Path::new(p).exists())
        .ok_or(AppError::NotFound)?;

    serve_file_response(audio_path, true).await
}

async fn processing_estimate(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> AppResult<Json<Value>> {
    let post = queries::get_post_by_guid(&state.db, &p_guid)
        .await?
        .ok_or(AppError::NotFound)?;

    let duration_sec = post.duration.unwrap_or(0) as f64;
    let estimate_sec = (duration_sec * 0.5).max(30.0);

    Ok(Json(json!({
        "estimate_seconds": estimate_sec,
        "duration_seconds": duration_sec,
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

    // Get transcript sample (first 10 segments)
    let segments = queries::get_segments_by_post(&state.db, post.id).await?;
    let transcript_sample: Vec<Value> = segments
        .iter()
        .take(10)
        .map(|s| {
            json!({
                "id": s.id,
                "sequence_num": s.sequence_num,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "text": s.text,
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
        "whitelisted": post.whitelisted,
        "download_count": post.download_count,
    })))
}

// Legacy routes
async fn serve_audio_legacy(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
) -> Result<Response, AppError> {
    serve_audio(State(state), AxumPath(p_guid)).await
}

async fn download_original_legacy(
    State(state): State<AppState>,
    AxumPath(p_guid): AxumPath<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    download_original(State(state), AxumPath(p_guid), auth_user).await
}

async fn serve_file_response(path: &str, as_attachment: bool) -> Result<Response, AppError> {
    let bytes = tokio::fs::read(path)
        .await
        .map_err(|_| AppError::NotFound)?;

    let filename = Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("audio.mp3");

    let content_type = if path.ends_with(".mp3") {
        "audio/mpeg"
    } else if path.ends_with(".ogg") {
        "audio/ogg"
    } else {
        "application/octet-stream"
    };

    let mut builder = axum::http::Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .header(axum::http::header::ACCEPT_RANGES, "bytes")
        .header(axum::http::header::CONTENT_LENGTH, bytes.len().to_string());

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

fn get_auth_user_id(auth_user: &Option<Extension<AuthenticatedUser>>) -> Option<i64> {
    auth_user.as_ref().map(|Extension(u)| u.id)
}
