use axum::extract::{Path, Query, State};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::middleware::require_admin_user;
use crate::auth::AuthenticatedUser;
use crate::db::queries;
use crate::error::AppResult;
use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/jobs/active", get(active_jobs))
        .route("/api/jobs/all", get(all_jobs))
        .route("/api/job-manager/status", get(job_manager_status))
        .route("/api/jobs/{job_id}/cancel", post(cancel_job))
        .route("/api/jobs/cancel-queued", post(cancel_queued_jobs))
        .route("/api/jobs/cleanup/preview", get(cleanup_preview))
        .route("/api/jobs/cleanup/run", post(cleanup_run))
}

#[derive(Deserialize)]
struct LimitQuery {
    limit: Option<i64>,
}

async fn active_jobs(
    State(state): State<AppState>,
    Query(q): Query<LimitQuery>,
) -> AppResult<Json<Value>> {
    let limit = q.limit.unwrap_or(100);
    let result = state.jobs_manager.list_active_jobs(limit).await;
    Ok(Json(result))
}

async fn all_jobs(
    State(state): State<AppState>,
    Query(q): Query<LimitQuery>,
) -> AppResult<Json<Value>> {
    let limit = q.limit.unwrap_or(100);
    let result = state.jobs_manager.list_all_jobs(limit).await;
    Ok(Json(result))
}

async fn job_manager_status(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let run: Option<crate::db::models::JobsManagerRun> = sqlx::query_as(
        "SELECT * FROM jobs_manager_run ORDER BY created_at DESC LIMIT 1",
    )
    .fetch_optional(&state.db)
    .await?;

    let run_json = if let Some(r) = run {
        // Compute live job counts like Python does
        let rid = &r.id;
        let total: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM processing_job WHERE jobs_manager_run_id = ?",
        )
        .bind(rid)
        .fetch_one(&state.db)
        .await
        .unwrap_or((0,));

        let queued: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM processing_job WHERE jobs_manager_run_id = ? AND status = 'pending'",
        )
        .bind(rid)
        .fetch_one(&state.db)
        .await
        .unwrap_or((0,));

        let running: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM processing_job WHERE jobs_manager_run_id = ? AND status = 'running'",
        )
        .bind(rid)
        .fetch_one(&state.db)
        .await
        .unwrap_or((0,));

        let completed: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM processing_job WHERE jobs_manager_run_id = ? AND status = 'completed'",
        )
        .bind(rid)
        .fetch_one(&state.db)
        .await
        .unwrap_or((0,));

        let failed: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM processing_job WHERE jobs_manager_run_id = ? AND status = 'failed'",
        )
        .bind(rid)
        .fetch_one(&state.db)
        .await
        .unwrap_or((0,));

        let skipped: (i64,) = sqlx::query_as(
            "SELECT COUNT(*) FROM processing_job WHERE jobs_manager_run_id = ? AND status = 'skipped'",
        )
        .bind(rid)
        .fetch_one(&state.db)
        .await
        .unwrap_or((0,));

        let progress = if total.0 > 0 {
            ((completed.0 + skipped.0) as f64 / total.0 as f64) * 100.0
        } else {
            0.0
        };

        json!({
            "id": r.id,
            "status": r.status,
            "trigger": r.trigger,
            "started_at": r.started_at,
            "completed_at": r.completed_at,
            "updated_at": r.updated_at,
            "total_jobs": total.0,
            "queued_jobs": queued.0,
            "running_jobs": running.0,
            "completed_jobs": completed.0,
            "failed_jobs": failed.0,
            "skipped_jobs": skipped.0,
            "context": r.context_json,
            "counters_reset_at": r.counters_reset_at,
            "progress_percentage": progress,
        })
    } else {
        Value::Null
    };

    Ok(Json(json!({ "run": run_json })))
}

async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<axum::response::Response, crate::error::AppError> {
    let (status_code, result) = state.jobs_manager.cancel_job(&job_id).await;
    let status = axum::http::StatusCode::from_u16(status_code).unwrap_or(axum::http::StatusCode::OK);
    Ok((status, Json(result)).into_response())
}

use axum::response::IntoResponse;

async fn cancel_queued_jobs(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let result = state.jobs_manager.cancel_queued_jobs().await;
    Ok(Json(result))
}

#[derive(Deserialize)]
struct CleanupQuery {
    retention_days: Option<i64>,
}

async fn cleanup_preview(
    State(state): State<AppState>,
    Query(q): Query<CleanupQuery>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let app = queries::get_app_settings(&state.db).await?;
    let retention_days = q.retention_days.or(app.post_cleanup_retention_days);

    if retention_days.is_none() || retention_days == Some(0) {
        return Ok(Json(json!({"count": 0, "retention_days": retention_days, "cutoff_utc": null})));
    }

    let days = retention_days.unwrap();
    let cutoff = chrono::Utc::now() - chrono::Duration::days(days);
    let cutoff_str = cutoff.to_rfc3339();

    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM post WHERE processed_audio_path IS NOT NULL AND release_date < ?",
    )
    .bind(&cutoff_str)
    .fetch_one(&state.db)
    .await?;

    Ok(Json(json!({
        "count": count.0, "retention_days": days, "cutoff_utc": cutoff_str,
    })))
}

async fn cleanup_run(
    State(state): State<AppState>,
    Query(q): Query<CleanupQuery>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let app = queries::get_app_settings(&state.db).await?;
    let retention_days = q.retention_days.or(app.post_cleanup_retention_days);

    if retention_days.is_none() || retention_days == Some(0) {
        return Ok(Json(json!({
            "status": "disabled", "message": "Cleanup is disabled because retention_days <= 0.",
        })));
    }

    let days = retention_days.unwrap();
    let cutoff = chrono::Utc::now() - chrono::Duration::days(days);
    let cutoff_str = cutoff.to_rfc3339();

    let posts: Vec<(i64, Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT id, processed_audio_path, unprocessed_audio_path FROM post WHERE processed_audio_path IS NOT NULL AND release_date < ?",
    )
    .bind(&cutoff_str)
    .fetch_all(&state.db)
    .await?;

    let removed = posts.len();

    for (post_id, processed, unprocessed) in &posts {
        if let Some(path) = processed {
            let _ = tokio::fs::remove_file(path).await;
        }
        if let Some(path) = unprocessed {
            let _ = tokio::fs::remove_file(path).await;
        }

        let _ = sqlx::query(
            "UPDATE post SET processed_audio_path = NULL, unprocessed_audio_path = NULL WHERE id = ?",
        )
        .bind(post_id)
        .execute(&state.db)
        .await;
    }

    // Count remaining candidates
    let remaining: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM post WHERE processed_audio_path IS NOT NULL AND release_date < ?",
    )
    .bind(&cutoff_str)
    .fetch_one(&state.db)
    .await
    .unwrap_or((0,));

    Ok(Json(json!({
        "status": "ok", "removed_posts": removed, "retention_days": days,
        "cutoff_utc": cutoff_str, "remaining_candidates": remaining.0,
    })))
}
