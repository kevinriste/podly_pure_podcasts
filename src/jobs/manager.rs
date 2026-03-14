use std::collections::HashSet;
use std::sync::Arc;

use serde_json::{json, Value};
use sqlx::SqlitePool;
use tokio::sync::{Mutex, Notify};

use crate::config::AppConfig;

/// Manages the lifecycle of processing jobs.
///
/// - Single background worker processes jobs sequentially (matches Python behavior)
/// - Per-post GUID locking prevents concurrent processing of the same post
/// - Jobs go through: pending → running → completed/failed/skipped/cancelled
pub struct JobsManager {
    pool: SqlitePool,
    config: Arc<AppConfig>,
    processing_locks: Mutex<HashSet<String>>,
    work_notify: Arc<Notify>,
    started: Mutex<bool>,
}

impl JobsManager {
    pub fn new(pool: SqlitePool, config: Arc<AppConfig>) -> Self {
        Self {
            pool,
            config,
            processing_locks: Mutex::new(HashSet::new()),
            work_notify: Arc::new(Notify::new()),
            started: Mutex::new(false),
        }
    }

    /// Start the background worker task.
    pub fn start(self: &Arc<Self>) {
        let mgr = Arc::clone(self);
        tokio::spawn(async move {
            let mut started = mgr.started.lock().await;
            if *started {
                return;
            }
            *started = true;
            drop(started);

            tracing::info!("JobsManager background worker started");
            mgr.worker_loop().await;
        });
    }

    async fn worker_loop(&self) {
        loop {
            // Try to dequeue and process the next pending job
            match self.dequeue_next_job().await {
                Some(job_id) => {
                    self.process_job(&job_id).await;
                }
                None => {
                    // Wait for new work (with 5-second timeout to recheck)
                    tokio::select! {
                        _ = self.work_notify.notified() => {},
                        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {},
                    }
                }
            }
        }
    }

    async fn dequeue_next_job(&self) -> Option<String> {
        // Atomically find the next pending job and mark it running
        let job: Option<(String, String)> = sqlx::query_as(
            "SELECT id, post_guid FROM processing_job WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .ok()?;

        let (job_id, post_guid) = job?;

        // Check per-post lock
        {
            let locks = self.processing_locks.lock().await;
            if locks.contains(&post_guid) {
                // Another job for this post is already running, skip for now
                return None;
            }
        }

        // Mark as running
        let now = chrono::Utc::now().to_rfc3339();
        let _ = sqlx::query(
            "UPDATE processing_job SET status = 'running', started_at = ?, step_name = 'starting' WHERE id = ? AND status = 'pending'",
        )
        .bind(&now)
        .bind(&job_id)
        .execute(&self.pool)
        .await;

        // Acquire per-post lock
        self.processing_locks.lock().await.insert(post_guid);

        Some(job_id)
    }

    async fn process_job(&self, job_id: &str) {
        let job: Option<(String, String, Option<i64>)> = sqlx::query_as(
            "SELECT id, post_guid, billing_user_id FROM processing_job WHERE id = ?",
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await
        .unwrap_or(None);

        let Some((_id, post_guid, _billing_user_id)) = job else {
            tracing::warn!("Job {job_id} disappeared before processing");
            return;
        };

        // Run the pipeline
        let result = super::pipeline::run_pipeline(&self.pool, &self.config, job_id, &post_guid).await;

        // Release per-post lock
        self.processing_locks.lock().await.remove(&post_guid);

        let now = chrono::Utc::now().to_rfc3339();
        match result {
            Ok(()) => {
                let _ = sqlx::query(
                    "UPDATE processing_job SET status = 'completed', completed_at = ?, progress_percentage = 100.0 WHERE id = ?",
                )
                .bind(&now)
                .bind(job_id)
                .execute(&self.pool)
                .await;
                tracing::info!("Job {job_id} completed for post {post_guid}");
            }
            Err(e) => {
                let err_msg = format!("{e}");
                let status = if err_msg.contains("cancelled") {
                    "cancelled"
                } else {
                    "failed"
                };
                let _ = sqlx::query(
                    "UPDATE processing_job SET status = ?, completed_at = ?, error_message = ? WHERE id = ?",
                )
                .bind(status)
                .bind(&now)
                .bind(&err_msg)
                .bind(job_id)
                .execute(&self.pool)
                .await;
                tracing::error!("Job {job_id} {status}: {err_msg}");
            }
        }
    }

    /// Start processing for a specific post. Returns status info.
    pub async fn start_post_processing(
        &self,
        post_guid: &str,
        requested_by_user_id: Option<i64>,
        billing_user_id: Option<i64>,
    ) -> Value {
        // Check if there's already an active job for this post
        let existing: Option<(String, String)> = sqlx::query_as(
            "SELECT id, status FROM processing_job WHERE post_guid = ? AND status IN ('pending', 'running') LIMIT 1",
        )
        .bind(post_guid)
        .fetch_optional(&self.pool)
        .await
        .unwrap_or(None);

        if let Some((existing_id, status)) = existing {
            return json!({
                "status": status,
                "message": "Job already exists",
                "job_id": existing_id,
            });
        }

        // Validate post
        let post: Option<(i64, String, bool, Option<String>)> = sqlx::query_as(
            "SELECT id, guid, whitelisted, download_url FROM post WHERE guid = ?",
        )
        .bind(post_guid)
        .fetch_optional(&self.pool)
        .await
        .unwrap_or(None);

        let Some((_post_id, _guid, whitelisted, download_url)) = post else {
            return json!({
                "status": "error",
                "error_code": "NOT_FOUND",
                "message": "Post not found",
            });
        };

        if !whitelisted {
            return json!({
                "status": "error",
                "error_code": "NOT_WHITELISTED",
                "message": "Post not whitelisted",
            });
        }

        if download_url.is_none() {
            return json!({
                "status": "error",
                "error_code": "NO_DOWNLOAD_URL",
                "message": "Post has no download URL",
            });
        }

        // Create the job
        let job_id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().to_rfc3339();

        let _ = sqlx::query(
            "INSERT INTO processing_job (id, post_guid, status, total_steps, created_at, requested_by_user_id, billing_user_id) VALUES (?, ?, 'pending', 6, ?, ?, ?)",
        )
        .bind(&job_id)
        .bind(post_guid)
        .bind(&now)
        .bind(requested_by_user_id)
        .bind(billing_user_id)
        .execute(&self.pool)
        .await;

        // Wake the worker
        self.work_notify.notify_one();

        json!({
            "status": "started",
            "message": "Processing job created",
            "job_id": job_id,
        })
    }

    /// List active (pending/running) jobs.
    pub async fn list_active_jobs(&self, limit: i64) -> Value {
        let jobs: Vec<crate::db::models::ProcessingJob> = sqlx::query_as(
            "SELECT * FROM processing_job WHERE status IN ('pending', 'running') ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        serde_json::to_value(jobs).unwrap_or(Value::Array(vec![]))
    }

    /// List all jobs with details.
    pub async fn list_all_jobs(&self, limit: i64) -> Value {
        let jobs: Vec<crate::db::models::ProcessingJob> = sqlx::query_as(
            "SELECT * FROM processing_job ORDER BY created_at DESC LIMIT ?",
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        serde_json::to_value(jobs).unwrap_or(Value::Array(vec![]))
    }

    /// Get processing status for a specific post.
    pub async fn get_post_status(&self, post_guid: &str) -> Value {
        // Check if post exists first
        let post: Option<(String, Option<String>)> = sqlx::query_as(
            "SELECT guid, processed_audio_path FROM post WHERE guid = ?",
        )
        .bind(post_guid)
        .fetch_optional(&self.pool)
        .await
        .unwrap_or(None);

        if post.is_none() {
            return json!({
                "status": "error",
                "error_code": "NOT_FOUND",
                "message": "Post not found",
            });
        }
        let (_guid, processed_path) = post.unwrap();

        // Check for processing job
        let job: Option<crate::db::models::ProcessingJob> = sqlx::query_as(
            "SELECT * FROM processing_job WHERE post_guid = ? ORDER BY created_at DESC LIMIT 1",
        )
        .bind(post_guid)
        .fetch_optional(&self.pool)
        .await
        .unwrap_or(None);

        if let Some(job) = job {
            let mut response = json!({
                "status": job.status,
                "step": job.current_step.unwrap_or(0),
                "step_name": job.step_name.as_deref().unwrap_or("Unknown"),
                "total_steps": job.total_steps.unwrap_or(4),
                "progress_percentage": job.progress_percentage.unwrap_or(0.0),
                "message": job.step_name.as_deref()
                    .unwrap_or(&format!("Step {} of {}", job.current_step.unwrap_or(0), job.total_steps.unwrap_or(4))),
            });
            if let Some(ref started_at) = job.started_at {
                response["started_at"] = json!(started_at);
            }
            if (job.status == "completed" || job.status == "skipped") && processed_path.is_some() {
                response["download_url"] = json!(format!("/api/posts/{post_guid}/download"));
            }
            if job.status == "failed" {
                if let Some(ref err) = job.error_message {
                    response["error"] = json!(err);
                }
            }
            if job.status == "cancelled" {
                if let Some(ref err) = job.error_message {
                    response["step_name"] = json!(err);
                    response["message"] = json!(err);
                }
            }
            return response;
        }

        // No job found
        if processed_path.is_some() {
            json!({
                "status": "skipped",
                "step": 4,
                "step_name": "Processing skipped",
                "total_steps": 4,
                "progress_percentage": 100.0,
                "message": "Post already processed",
                "download_url": format!("/api/posts/{post_guid}/download"),
            })
        } else {
            json!({
                "status": "not_started",
                "step": 0,
                "step_name": "Not started",
                "total_steps": 4,
                "progress_percentage": 0.0,
                "message": "No processing job found",
            })
        }
    }

    /// Cancel a specific job.
    pub async fn cancel_job(&self, job_id: &str) -> Value {
        let job: Option<(String, String)> = sqlx::query_as(
            "SELECT id, status FROM processing_job WHERE id = ?",
        )
        .bind(job_id)
        .fetch_optional(&self.pool)
        .await
        .unwrap_or(None);

        let Some((id, status)) = job else {
            return json!({
                "status": "error",
                "error_code": "NOT_FOUND",
                "message": "Job not found",
            });
        };

        if status != "pending" && status != "running" {
            return json!({
                "status": "error",
                "error_code": "NOT_CANCELLABLE",
                "message": format!("Job is already {status}"),
            });
        }

        let now = chrono::Utc::now().to_rfc3339();
        let _ = sqlx::query(
            "UPDATE processing_job SET status = 'cancelled', completed_at = ? WHERE id = ?",
        )
        .bind(&now)
        .bind(&id)
        .execute(&self.pool)
        .await;

        json!({
            "status": "cancelled",
            "job_id": id,
            "message": "Job cancelled",
        })
    }

    /// Cancel all queued (pending) jobs.
    pub async fn cancel_queued_jobs(&self) -> Value {
        let now = chrono::Utc::now().to_rfc3339();
        let result = sqlx::query(
            "UPDATE processing_job SET status = 'cancelled', completed_at = ? WHERE status = 'pending'",
        )
        .bind(&now)
        .execute(&self.pool)
        .await;

        let cancelled = result.map(|r| r.rows_affected()).unwrap_or(0);

        json!({
            "status": "ok",
            "cancelled_count": cancelled,
        })
    }

    /// Cancel all jobs for a specific post.
    pub async fn cancel_post_jobs(&self, post_guid: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let _ = sqlx::query(
            "UPDATE processing_job SET status = 'cancelled', completed_at = ? WHERE post_guid = ? AND status IN ('pending', 'running')",
        )
        .bind(&now)
        .bind(post_guid)
        .execute(&self.pool)
        .await;
    }

    /// Enqueue pending jobs for all whitelisted posts that need processing.
    pub async fn enqueue_pending_jobs(&self) -> i64 {
        // Find whitelisted posts without processed audio and without active jobs
        let posts: Vec<(String,)> = sqlx::query_as(
            "SELECT p.guid FROM post p
             WHERE p.whitelisted = 1
             AND p.processed_audio_path IS NULL
             AND p.download_url IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM processing_job j
                 WHERE j.post_guid = p.guid
                 AND j.status IN ('pending', 'running')
             )
             ORDER BY p.release_date DESC",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut enqueued = 0i64;
        for (guid,) in &posts {
            let job_id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            let result = sqlx::query(
                "INSERT INTO processing_job (id, post_guid, status, total_steps, created_at) VALUES (?, ?, 'pending', 6, ?)",
            )
            .bind(&job_id)
            .bind(guid)
            .bind(&now)
            .execute(&self.pool)
            .await;

            if result.is_ok() {
                enqueued += 1;
            }
        }

        if enqueued > 0 {
            self.work_notify.notify_one();
            tracing::info!("Enqueued {enqueued} pending processing jobs");
        }

        enqueued
    }

    /// Refresh all feeds and enqueue pending jobs.
    pub async fn start_refresh_all_feeds(&self) -> Value {
        let feeds: Vec<(i64, String)> =
            sqlx::query_as("SELECT id, rss_url FROM feed")
                .fetch_all(&self.pool)
                .await
                .unwrap_or_default();

        let feed_count = feeds.len();
        let mut errors = Vec::new();

        for (feed_id, rss_url) in &feeds {
            if let Err(e) = crate::feeds::refresh::refresh_feed(&self.pool, *feed_id, rss_url).await
            {
                tracing::error!("Failed to refresh feed {feed_id}: {e}");
                errors.push(format!("Feed {feed_id}: {e}"));
            }
        }

        let enqueued = self.enqueue_pending_jobs().await;

        json!({
            "status": "success",
            "feeds_refreshed": feed_count,
            "enqueued": enqueued,
            "errors": errors,
        })
    }
}
