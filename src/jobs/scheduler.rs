use std::sync::Arc;

use super::manager::JobsManager;

/// Start the background scheduler for periodic tasks.
///
/// Tasks:
/// - Feed refresh: check all feeds for new episodes based on configured interval
/// - Auto-processing: enqueue jobs for whitelisted unprocessed posts
pub async fn start_scheduler(
    jobs_manager: Arc<JobsManager>,
    pool: sqlx::SqlitePool,
) {
    tokio::spawn(async move {
        loop {
            // Get configured interval (default 60 minutes)
            let interval_minutes = get_refresh_interval(&pool).await;

            if interval_minutes > 0 {
                tracing::info!("Scheduler: sleeping for {interval_minutes} minutes");
                tokio::time::sleep(tokio::time::Duration::from_secs(interval_minutes as u64 * 60)).await;

                tracing::info!("Scheduler: starting periodic feed refresh");

                // Refresh all feeds
                jobs_manager.start_refresh_all_feeds().await;

                // Enqueue pending jobs for newly whitelisted posts
                let enqueued = jobs_manager.enqueue_pending_jobs().await;
                if enqueued > 0 {
                    tracing::info!("Scheduler: enqueued {enqueued} new processing jobs");
                }
            } else {
                // Disabled — check again in 5 minutes
                tokio::time::sleep(tokio::time::Duration::from_secs(300)).await;
            }
        }
    });
}

async fn get_refresh_interval(pool: &sqlx::SqlitePool) -> i64 {
    let result: Option<(Option<i64>,)> = sqlx::query_as(
        "SELECT background_update_interval_minute FROM app_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    result.and_then(|(v,)| v).unwrap_or(60)
}
