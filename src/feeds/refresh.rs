use sqlx::SqlitePool;

use super::parser;

/// Refresh a feed by fetching its RSS URL and inserting any new episodes.
pub async fn refresh_feed(
    pool: &SqlitePool,
    feed_id: i64,
    rss_url: &str,
) -> Result<RefreshResult, RefreshError> {
    let parsed = parser::fetch_and_parse(rss_url)
        .await
        .map_err(|e| RefreshError::Parse(e.to_string()))?;

    // Get existing post GUIDs for this feed
    let existing: Vec<(String,)> =
        sqlx::query_as("SELECT guid FROM post WHERE feed_id = ?")
            .bind(feed_id)
            .fetch_all(pool)
            .await
            .map_err(|e| RefreshError::Db(e.to_string()))?;

    let existing_guids: std::collections::HashSet<String> =
        existing.into_iter().map(|(g,)| g).collect();

    // Get app settings for auto-whitelist behavior
    let app_settings: Option<(bool, i64)> = sqlx::query_as(
        "SELECT automatically_whitelist_new_episodes, number_of_episodes_to_whitelist_from_archive_of_new_feed FROM app_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    let auto_whitelist = app_settings.map(|(a, _)| a).unwrap_or(false);

    // Get feed-level override
    let feed_override: Option<(Option<bool>,)> = sqlx::query_as(
        "SELECT auto_whitelist_new_episodes_override FROM feed WHERE id = ?",
    )
    .bind(feed_id)
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    let feed_auto_override = feed_override.and_then(|(o,)| o);

    // Check if this is a brand new feed (no existing posts)
    let is_new_feed = existing_guids.is_empty();

    let mut new_count = 0;
    for (i, episode) in parsed.episodes.iter().enumerate() {
        if existing_guids.contains(&episode.guid) {
            continue;
        }

        // Determine whitelist status
        let should_whitelist = if is_new_feed {
            // For new feeds, only whitelist the most recent episodes
            let archive_count = app_settings.map(|(_, c)| c).unwrap_or(1) as usize;
            i < archive_count
        } else {
            // For existing feeds, use feed override or global setting
            // But don't auto-whitelist back-catalog episodes on refresh
            let is_recent = episode.release_date.map_or(false, |d| {
                let cutoff = chrono::Utc::now() - chrono::Duration::days(7);
                d > cutoff
            });

            if !is_recent {
                false
            } else {
                feed_auto_override.unwrap_or(auto_whitelist)
            }
        };

        let _now = chrono::Utc::now().to_rfc3339();
        let release_date = episode
            .release_date
            .map(|d| d.to_rfc3339())
            .unwrap_or_default();

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO post (feed_id, guid, title, description, download_url, release_date, duration, whitelisted, image_url)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(feed_id)
        .bind(&episode.guid)
        .bind(&episode.title)
        .bind(&episode.description)
        .bind(&episode.audio_url)
        .bind(&release_date)
        .bind(episode.duration)
        .bind(should_whitelist)
        .bind(&episode.image_url)
        .execute(pool)
        .await;

        new_count += 1;
    }

    // Update feed metadata if changed
    if !parsed.title.is_empty() {
        let _ = sqlx::query(
            "UPDATE feed SET title = ?, description = ?, author = ?, image_url = COALESCE(?, image_url) WHERE id = ?",
        )
        .bind(&parsed.title)
        .bind(&parsed.description)
        .bind(&parsed.author)
        .bind(&parsed.image_url)
        .bind(feed_id)
        .execute(pool)
        .await;
    }

    Ok(RefreshResult {
        new_episodes: new_count,
        total_episodes: parsed.episodes.len(),
    })
}

pub struct RefreshResult {
    pub new_episodes: usize,
    pub total_episodes: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RefreshError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("database error: {0}")]
    Db(String),
}
