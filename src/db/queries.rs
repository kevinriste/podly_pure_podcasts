use sqlx::SqlitePool;

use super::models::*;
use crate::error::AppResult;

// ── Feed queries ──

pub async fn get_all_feeds(pool: &SqlitePool) -> AppResult<Vec<Feed>> {
    let feeds = sqlx::query_as::<_, Feed>("SELECT * FROM feed ORDER BY title")
        .fetch_all(pool)
        .await?;
    Ok(feeds)
}

pub async fn get_feed_by_id(pool: &SqlitePool, id: i64) -> AppResult<Option<Feed>> {
    let feed = sqlx::query_as::<_, Feed>("SELECT * FROM feed WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(feed)
}

pub async fn get_feed_by_rss_url(pool: &SqlitePool, rss_url: &str) -> AppResult<Option<Feed>> {
    let feed = sqlx::query_as::<_, Feed>("SELECT * FROM feed WHERE rss_url = ?")
        .bind(rss_url)
        .fetch_optional(pool)
        .await?;
    Ok(feed)
}

pub async fn insert_feed(
    pool: &SqlitePool,
    title: &str,
    rss_url: &str,
    description: Option<&str>,
    author: Option<&str>,
    image_url: Option<&str>,
) -> AppResult<i64> {
    let result = sqlx::query(
        "INSERT INTO feed (title, rss_url, description, author, image_url) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(title)
    .bind(rss_url)
    .bind(description)
    .bind(author)
    .bind(image_url)
    .execute(pool)
    .await?;
    Ok(result.last_insert_rowid())
}

pub async fn delete_feed(pool: &SqlitePool, id: i64) -> AppResult<()> {
    sqlx::query("DELETE FROM feed WHERE id = ?")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn delete_feed_cascade(pool: &SqlitePool, feed_id: i64) -> AppResult<Vec<(Option<String>, Option<String>)>> {
    // Get audio paths before deleting
    let paths: Vec<(Option<String>, Option<String>)> = sqlx::query_as(
        "SELECT processed_audio_path, unprocessed_audio_path FROM post WHERE feed_id = ?",
    )
    .bind(feed_id)
    .fetch_all(pool)
    .await?;

    // Get post IDs for cascading deletes
    let post_ids: Vec<(i64,)> = sqlx::query_as("SELECT id FROM post WHERE feed_id = ?")
        .bind(feed_id)
        .fetch_all(pool)
        .await?;

    for (post_id,) in &post_ids {
        // Delete identifications via segments
        sqlx::query(
            "DELETE FROM identification WHERE transcript_segment_id IN (SELECT id FROM transcript_segment WHERE post_id = ?)",
        )
        .bind(post_id)
        .execute(pool)
        .await?;

        sqlx::query("DELETE FROM transcript_segment WHERE post_id = ?")
            .bind(post_id)
            .execute(pool)
            .await?;

        sqlx::query("DELETE FROM model_call WHERE post_id = ?")
            .bind(post_id)
            .execute(pool)
            .await?;
    }

    // Delete processing jobs for this feed's posts
    sqlx::query(
        "DELETE FROM processing_job WHERE post_guid IN (SELECT guid FROM post WHERE feed_id = ?)",
    )
    .bind(feed_id)
    .execute(pool)
    .await?;

    sqlx::query("DELETE FROM post WHERE feed_id = ?")
        .bind(feed_id)
        .execute(pool)
        .await?;

    sqlx::query("DELETE FROM feed_supporter WHERE feed_id = ?")
        .bind(feed_id)
        .execute(pool)
        .await?;

    sqlx::query("DELETE FROM feed_access_token WHERE feed_id = ?")
        .bind(feed_id)
        .execute(pool)
        .await?;

    sqlx::query("DELETE FROM feed WHERE id = ?")
        .bind(feed_id)
        .execute(pool)
        .await?;

    Ok(paths)
}

pub async fn update_feed_metadata(
    pool: &SqlitePool,
    feed_id: i64,
    title: &str,
    description: Option<&str>,
    author: Option<&str>,
    image_url: Option<&str>,
) -> AppResult<()> {
    sqlx::query("UPDATE feed SET title = ?, description = ?, author = ?, image_url = ? WHERE id = ?")
        .bind(title)
        .bind(description)
        .bind(author)
        .bind(image_url)
        .bind(feed_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_feed_strategy(pool: &SqlitePool, feed_id: i64, strategy: &str) -> AppResult<()> {
    sqlx::query("UPDATE feed SET ad_detection_strategy = ? WHERE id = ?")
        .bind(strategy)
        .bind(feed_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_feed_chapter_filter(pool: &SqlitePool, feed_id: i64, filter: Option<&str>) -> AppResult<()> {
    sqlx::query("UPDATE feed SET chapter_filter_strings = ? WHERE id = ?")
        .bind(filter)
        .bind(feed_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_feed_auto_whitelist(pool: &SqlitePool, feed_id: i64, override_val: Option<bool>) -> AppResult<()> {
    sqlx::query("UPDATE feed SET auto_whitelist_new_episodes_override = ? WHERE id = ?")
        .bind(override_val)
        .bind(feed_id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Post queries ──

pub async fn get_posts_by_feed(
    pool: &SqlitePool,
    feed_id: i64,
    page: i64,
    page_size: i64,
    whitelisted_only: bool,
) -> AppResult<(Vec<Post>, i64)> {
    let offset = (page - 1) * page_size;

    let (posts, total) = if whitelisted_only {
        let posts = sqlx::query_as::<_, Post>(
            "SELECT * FROM post WHERE feed_id = ? AND whitelisted = 1 ORDER BY release_date DESC NULLS LAST, id DESC LIMIT ? OFFSET ?",
        )
        .bind(feed_id)
        .bind(page_size)
        .bind(offset)
        .fetch_all(pool)
        .await?;

        let total: (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM post WHERE feed_id = ? AND whitelisted = 1")
                .bind(feed_id)
                .fetch_one(pool)
                .await?;

        (posts, total.0)
    } else {
        let posts = sqlx::query_as::<_, Post>(
            "SELECT * FROM post WHERE feed_id = ? ORDER BY release_date DESC NULLS LAST, id DESC LIMIT ? OFFSET ?",
        )
        .bind(feed_id)
        .bind(page_size)
        .bind(offset)
        .fetch_all(pool)
        .await?;

        let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM post WHERE feed_id = ?")
            .bind(feed_id)
            .fetch_one(pool)
            .await?;

        (posts, total.0)
    };

    Ok((posts, total))
}

pub async fn get_post_by_guid(pool: &SqlitePool, guid: &str) -> AppResult<Option<Post>> {
    let post = sqlx::query_as::<_, Post>("SELECT * FROM post WHERE guid = ?")
        .bind(guid)
        .fetch_optional(pool)
        .await?;
    Ok(post)
}

pub async fn get_whitelisted_total(pool: &SqlitePool, feed_id: i64) -> AppResult<i64> {
    let count: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM post WHERE feed_id = ? AND whitelisted = 1")
            .bind(feed_id)
            .fetch_one(pool)
            .await?;
    Ok(count.0)
}

pub async fn get_whitelisted_posts_for_feed(pool: &SqlitePool, feed_id: i64) -> AppResult<Vec<Post>> {
    // Python parity: only include whitelisted posts that have been processed
    let posts = sqlx::query_as::<_, Post>(
        "SELECT * FROM post WHERE feed_id = ? AND whitelisted = 1 AND processed_audio_path IS NOT NULL ORDER BY release_date DESC",
    )
    .bind(feed_id)
    .fetch_all(pool)
    .await?;
    Ok(posts)
}

pub async fn get_all_posts_for_feed(pool: &SqlitePool, feed_id: i64) -> AppResult<Vec<Post>> {
    let posts = sqlx::query_as::<_, Post>(
        "SELECT * FROM post WHERE feed_id = ? ORDER BY release_date DESC",
    )
    .bind(feed_id)
    .fetch_all(pool)
    .await?;
    Ok(posts)
}

pub async fn insert_post(
    pool: &SqlitePool,
    feed_id: i64,
    guid: &str,
    title: &str,
    download_url: &str,
    description: Option<&str>,
    release_date: Option<&str>,
    duration: Option<i64>,
    image_url: Option<&str>,
    chapter_data: Option<&str>,
    whitelisted: bool,
) -> AppResult<i64> {
    let result = sqlx::query(
        "INSERT OR IGNORE INTO post (feed_id, guid, title, download_url, description, release_date, duration, image_url, chapter_data, whitelisted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(feed_id)
    .bind(guid)
    .bind(title)
    .bind(download_url)
    .bind(description)
    .bind(release_date)
    .bind(duration)
    .bind(image_url)
    .bind(chapter_data)
    .bind(whitelisted)
    .execute(pool)
    .await?;
    Ok(result.last_insert_rowid())
}

pub async fn set_post_whitelist(pool: &SqlitePool, post_id: i64, whitelisted: bool) -> AppResult<()> {
    sqlx::query("UPDATE post SET whitelisted = ? WHERE id = ?")
        .bind(whitelisted)
        .bind(post_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn are_all_posts_whitelisted(pool: &SqlitePool, feed_id: i64) -> AppResult<bool> {
    let total: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM post WHERE feed_id = ?")
        .bind(feed_id)
        .fetch_one(pool)
        .await?;
    let whitelisted: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM post WHERE feed_id = ? AND whitelisted = 1")
        .bind(feed_id)
        .fetch_one(pool)
        .await?;
    Ok(total.0 > 0 && total.0 == whitelisted.0)
}

pub async fn set_all_posts_whitelist(pool: &SqlitePool, feed_id: i64, whitelisted: bool) -> AppResult<u64> {
    let result = sqlx::query("UPDATE post SET whitelisted = ? WHERE feed_id = ?")
        .bind(whitelisted)
        .bind(feed_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

pub async fn increment_download_count(pool: &SqlitePool, post_id: i64) -> AppResult<()> {
    sqlx::query("UPDATE post SET download_count = COALESCE(download_count, 0) + 1 WHERE id = ?")
        .bind(post_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn get_recent_processed_posts(pool: &SqlitePool, feed_id: i64, limit: i64) -> AppResult<Vec<Post>> {
    let posts = sqlx::query_as::<_, Post>(
        "SELECT p.*, f.title as feed_title FROM post p JOIN feed f ON p.feed_id = f.id WHERE p.feed_id = ? AND p.whitelisted = 1 AND p.processed_audio_path IS NOT NULL ORDER BY p.release_date DESC LIMIT ?",
    )
    .bind(feed_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(posts)
}

pub async fn clear_post_processing_data(pool: &SqlitePool, post_id: i64) -> AppResult<()> {
    // Delete identifications
    sqlx::query(
        "DELETE FROM identification WHERE transcript_segment_id IN (SELECT id FROM transcript_segment WHERE post_id = ?)",
    )
    .bind(post_id)
    .execute(pool)
    .await?;

    // Delete transcript segments
    sqlx::query("DELETE FROM transcript_segment WHERE post_id = ?")
        .bind(post_id)
        .execute(pool)
        .await?;

    // Delete model calls
    sqlx::query("DELETE FROM model_call WHERE post_id = ?")
        .bind(post_id)
        .execute(pool)
        .await?;

    // Delete processing jobs for this post
    sqlx::query(
        "DELETE FROM processing_job WHERE post_guid = (SELECT guid FROM post WHERE id = ?)",
    )
    .bind(post_id)
    .execute(pool)
    .await?;

    // Clear processed audio path, refined boundaries, and reset duration
    sqlx::query(
        "UPDATE post SET processed_audio_path = NULL, refined_ad_boundaries = NULL, refined_ad_boundaries_updated_at = NULL, duration = NULL WHERE id = ?",
    )
    .bind(post_id)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn clear_post_identifications(pool: &SqlitePool, post_id: i64) -> AppResult<()> {
    sqlx::query(
        "DELETE FROM identification WHERE transcript_segment_id IN (SELECT id FROM transcript_segment WHERE post_id = ?)",
    )
    .bind(post_id)
    .execute(pool)
    .await?;

    // Python parity: preserve Whisper model calls so transcript can be reused
    sqlx::query("DELETE FROM model_call WHERE post_id = ? AND model_name NOT LIKE '%whisper%'")
        .bind(post_id)
        .execute(pool)
        .await?;

    sqlx::query(
        "UPDATE post SET processed_audio_path = NULL, refined_ad_boundaries = NULL, refined_ad_boundaries_updated_at = NULL WHERE id = ?",
    )
    .bind(post_id)
    .execute(pool)
    .await?;

    Ok(())
}

// ── User queries ──

pub async fn get_user_by_username(pool: &SqlitePool, username: &str) -> AppResult<Option<User>> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE username = ?")
        .bind(username.to_lowercase().trim())
        .fetch_optional(pool)
        .await?;
    Ok(user)
}

pub async fn get_user_by_id(pool: &SqlitePool, id: i64) -> AppResult<Option<User>> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(user)
}

pub async fn get_all_users(pool: &SqlitePool) -> AppResult<Vec<User>> {
    let users = sqlx::query_as::<_, User>("SELECT * FROM users ORDER BY created_at DESC, id DESC")
        .fetch_all(pool)
        .await?;
    Ok(users)
}

pub async fn insert_user(
    pool: &SqlitePool,
    username: &str,
    password_hash: &str,
    role: &str,
) -> AppResult<i64> {
    let now = chrono::Utc::now().to_rfc3339();
    let result = sqlx::query(
        "INSERT INTO users (username, password_hash, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(username.to_lowercase().trim())
    .bind(password_hash)
    .bind(role)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(result.last_insert_rowid())
}

pub async fn count_users(pool: &SqlitePool) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM users")
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

pub async fn count_admin_users(pool: &SqlitePool) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

pub async fn update_user_last_active(pool: &SqlitePool, user_id: i64) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("UPDATE users SET last_active = ? WHERE id = ?")
        .bind(&now)
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_user_password(pool: &SqlitePool, user_id: i64, password_hash: &str) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?")
        .bind(password_hash)
        .bind(&now)
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_user_role(pool: &SqlitePool, user_id: i64, role: &str) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("UPDATE users SET role = ?, updated_at = ? WHERE id = ?")
        .bind(role)
        .bind(&now)
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

pub async fn update_user_manual_feed_allowance(pool: &SqlitePool, user_id: i64, allowance: Option<i64>) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query("UPDATE users SET manual_feed_allowance = ?, updated_at = ? WHERE id = ?")
        .bind(allowance)
        .bind(&now)
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Discord user queries ──

pub async fn get_user_by_discord_id(pool: &SqlitePool, discord_id: &str) -> AppResult<Option<User>> {
    let user = sqlx::query_as::<_, User>("SELECT * FROM users WHERE discord_id = ?")
        .bind(discord_id)
        .fetch_optional(pool)
        .await?;
    Ok(user)
}

pub async fn upsert_discord_user(
    pool: &SqlitePool,
    discord_id: &str,
    discord_username: &str,
) -> AppResult<(i64, bool)> {
    // Check existing
    if let Some(existing) = get_user_by_discord_id(pool, discord_id).await? {
        let now = chrono::Utc::now().to_rfc3339();
        sqlx::query("UPDATE users SET discord_username = ?, updated_at = ? WHERE id = ?")
            .bind(discord_username)
            .bind(&now)
            .bind(existing.id)
            .execute(pool)
            .await?;
        return Ok((existing.id, false));
    }

    // Generate unique username from discord name
    let base = discord_username.to_lowercase().replace(' ', "_");
    let base = if base.len() > 50 { &base[..50] } else { &base };
    let mut username = base.to_string();
    let mut counter = 1u32;
    while get_user_by_username(pool, &username).await?.is_some() {
        username = format!("{base}_{counter}");
        counter += 1;
    }

    let now = chrono::Utc::now().to_rfc3339();
    let result = sqlx::query(
        "INSERT INTO users (username, password_hash, role, discord_id, discord_username, created_at, updated_at) VALUES (?, '', 'user', ?, ?, ?, ?)",
    )
    .bind(&username)
    .bind(discord_id)
    .bind(discord_username)
    .bind(&now)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok((result.last_insert_rowid(), true))
}

pub async fn update_discord_settings(
    pool: &SqlitePool,
    client_id: Option<&str>,
    client_secret: Option<&str>,
    redirect_uri: Option<&str>,
    guild_ids: Option<&str>,
    allow_registration: bool,
) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "UPDATE discord_settings SET client_id = ?, client_secret = ?, redirect_uri = ?, guild_ids = ?, allow_registration = ?, updated_at = ? WHERE id = 1",
    )
    .bind(client_id)
    .bind(client_secret)
    .bind(redirect_uri)
    .bind(guild_ids)
    .bind(allow_registration)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

// ── Billing queries ──

pub async fn set_user_billing_fields(
    pool: &SqlitePool,
    user_id: i64,
    stripe_customer_id: Option<&str>,
    stripe_subscription_id: Option<&str>,
    feed_allowance: Option<i64>,
    feed_subscription_status: Option<&str>,
) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    // Build dynamic update
    let mut sets = vec!["updated_at = ?"];
    if stripe_customer_id.is_some() { sets.push("stripe_customer_id = ?"); }
    if stripe_subscription_id.is_some() { sets.push("stripe_subscription_id = ?"); }
    if feed_allowance.is_some() { sets.push("feed_allowance = ?"); }
    if feed_subscription_status.is_some() { sets.push("feed_subscription_status = ?"); }

    // Use a simpler approach — always set all fields
    sqlx::query(
        "UPDATE users SET stripe_customer_id = COALESCE(?, stripe_customer_id), stripe_subscription_id = COALESCE(?, stripe_subscription_id), feed_allowance = COALESCE(?, feed_allowance), feed_subscription_status = COALESCE(?, feed_subscription_status), updated_at = ? WHERE id = ?",
    )
    .bind(stripe_customer_id)
    .bind(stripe_subscription_id)
    .bind(feed_allowance)
    .bind(feed_subscription_status)
    .bind(&now)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn set_user_billing_by_customer_id(
    pool: &SqlitePool,
    stripe_customer_id: &str,
    stripe_subscription_id: Option<&str>,
    feed_allowance: i64,
    feed_subscription_status: &str,
) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "UPDATE users SET stripe_subscription_id = COALESCE(?, stripe_subscription_id), feed_allowance = ?, feed_subscription_status = ?, updated_at = ? WHERE stripe_customer_id = ?",
    )
    .bind(stripe_subscription_id)
    .bind(feed_allowance)
    .bind(feed_subscription_status)
    .bind(&now)
    .bind(stripe_customer_id)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn delete_user(pool: &SqlitePool, user_id: i64) -> AppResult<()> {
    // Clean up user's feed memberships and tokens
    sqlx::query("DELETE FROM feed_supporter WHERE user_id = ?")
        .bind(user_id)
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM feed_access_token WHERE user_id = ?")
        .bind(user_id)
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM users WHERE id = ?")
        .bind(user_id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Settings queries ──

pub async fn get_llm_settings(pool: &SqlitePool) -> AppResult<LlmSettings> {
    let settings = sqlx::query_as::<_, LlmSettings>("SELECT * FROM llm_settings WHERE id = 1")
        .fetch_one(pool)
        .await?;
    Ok(settings)
}

pub async fn get_whisper_settings(pool: &SqlitePool) -> AppResult<WhisperSettings> {
    let settings =
        sqlx::query_as::<_, WhisperSettings>("SELECT * FROM whisper_settings WHERE id = 1")
            .fetch_one(pool)
            .await?;
    Ok(settings)
}

pub async fn get_processing_settings(pool: &SqlitePool) -> AppResult<ProcessingSettings> {
    let settings =
        sqlx::query_as::<_, ProcessingSettings>("SELECT * FROM processing_settings WHERE id = 1")
            .fetch_one(pool)
            .await?;
    Ok(settings)
}

pub async fn get_output_settings(pool: &SqlitePool) -> AppResult<OutputSettings> {
    let settings =
        sqlx::query_as::<_, OutputSettings>("SELECT * FROM output_settings WHERE id = 1")
            .fetch_one(pool)
            .await?;
    Ok(settings)
}

pub async fn get_app_settings(pool: &SqlitePool) -> AppResult<AppSettings> {
    let settings = sqlx::query_as::<_, AppSettings>("SELECT * FROM app_settings WHERE id = 1")
        .fetch_one(pool)
        .await?;
    Ok(settings)
}

pub async fn get_discord_settings(pool: &SqlitePool) -> AppResult<DiscordSettings> {
    let settings =
        sqlx::query_as::<_, DiscordSettings>("SELECT * FROM discord_settings WHERE id = 1")
            .fetch_one(pool)
            .await?;
    Ok(settings)
}

pub async fn get_chapter_filter_settings(pool: &SqlitePool) -> AppResult<ChapterFilterSettings> {
    let settings = sqlx::query_as::<_, ChapterFilterSettings>(
        "SELECT * FROM chapter_filter_settings WHERE id = 1",
    )
    .fetch_one(pool)
    .await?;
    Ok(settings)
}

// ── UserFeed / feed_supporter queries ──

pub async fn get_user_feeds(pool: &SqlitePool, user_id: i64) -> AppResult<Vec<UserFeed>> {
    let feeds = sqlx::query_as::<_, UserFeed>("SELECT * FROM feed_supporter WHERE user_id = ?")
        .bind(user_id)
        .fetch_all(pool)
        .await?;
    Ok(feeds)
}

pub async fn count_user_feeds(pool: &SqlitePool, user_id: i64) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM feed_supporter WHERE user_id = ?")
        .bind(user_id)
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

pub async fn get_feed_member_count(pool: &SqlitePool, feed_id: i64) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM feed_supporter WHERE feed_id = ?")
        .bind(feed_id)
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

pub async fn get_user_visible_feeds(pool: &SqlitePool, user_id: i64) -> AppResult<Vec<Feed>> {
    // Users can see feeds they've joined plus feed id=1 (the default feed)
    let feeds = sqlx::query_as::<_, Feed>(
        "SELECT DISTINCT f.* FROM feed f \
         LEFT JOIN feed_supporter fs ON f.id = fs.feed_id AND fs.user_id = ? \
         WHERE fs.id IS NOT NULL OR f.id = 1 \
         ORDER BY f.title",
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(feeds)
}

pub async fn is_feed_member(pool: &SqlitePool, user_id: i64, feed_id: i64) -> AppResult<bool> {
    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM feed_supporter WHERE user_id = ? AND feed_id = ?",
    )
    .bind(user_id)
    .bind(feed_id)
    .fetch_one(pool)
    .await?;
    Ok(count.0 > 0)
}

pub async fn ensure_feed_membership(pool: &SqlitePool, user_id: i64, feed_id: i64) -> AppResult<()> {
    let now = chrono::Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT OR IGNORE INTO feed_supporter (user_id, feed_id, created_at) VALUES (?, ?, ?)",
    )
    .bind(user_id)
    .bind(feed_id)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn remove_feed_membership(pool: &SqlitePool, user_id: i64, feed_id: i64) -> AppResult<()> {
    sqlx::query("DELETE FROM feed_supporter WHERE user_id = ? AND feed_id = ?")
        .bind(user_id)
        .bind(feed_id)
        .execute(pool)
        .await?;
    Ok(())
}

// ── Feed access token queries ──

pub async fn insert_feed_access_token(
    pool: &SqlitePool,
    token_id: &str,
    token_hash: &str,
    token_secret: &str,
    feed_id: Option<i64>,
    user_id: i64,
) -> AppResult<i64> {
    let now = chrono::Utc::now().to_rfc3339();
    let result = sqlx::query(
        "INSERT INTO feed_access_token (token_id, token_hash, token_secret, feed_id, user_id, created_at, revoked) VALUES (?, ?, ?, ?, ?, ?, 0)",
    )
    .bind(token_id)
    .bind(token_hash)
    .bind(token_secret)
    .bind(feed_id)
    .bind(user_id)
    .bind(&now)
    .execute(pool)
    .await?;
    Ok(result.last_insert_rowid())
}

pub async fn get_feed_access_token_by_id(pool: &SqlitePool, token_id: &str) -> AppResult<Option<FeedAccessToken>> {
    let token = sqlx::query_as::<_, FeedAccessToken>(
        "SELECT * FROM feed_access_token WHERE token_id = ?",
    )
    .bind(token_id)
    .fetch_optional(pool)
    .await?;
    Ok(token)
}

// ── TranscriptSegment queries ──

pub async fn get_segments_by_post(
    pool: &SqlitePool,
    post_id: i64,
) -> AppResult<Vec<TranscriptSegment>> {
    let segments = sqlx::query_as::<_, TranscriptSegment>(
        "SELECT * FROM transcript_segment WHERE post_id = ? ORDER BY sequence_num",
    )
    .bind(post_id)
    .fetch_all(pool)
    .await?;
    Ok(segments)
}

pub async fn count_segments_by_post(pool: &SqlitePool, post_id: i64) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM transcript_segment WHERE post_id = ?")
        .bind(post_id)
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

// ── Identification queries ──

pub async fn get_identifications_by_post(pool: &SqlitePool, post_id: i64) -> AppResult<Vec<Identification>> {
    let ids = sqlx::query_as::<_, Identification>(
        "SELECT i.* FROM identification i \
         JOIN transcript_segment ts ON i.transcript_segment_id = ts.id \
         WHERE ts.post_id = ? \
         ORDER BY ts.sequence_num",
    )
    .bind(post_id)
    .fetch_all(pool)
    .await?;
    Ok(ids)
}

// ── ModelCall queries ──

pub async fn get_model_calls_by_post(pool: &SqlitePool, post_id: i64) -> AppResult<Vec<ModelCall>> {
    let calls = sqlx::query_as::<_, ModelCall>(
        "SELECT * FROM model_call WHERE post_id = ? ORDER BY first_segment_sequence_num",
    )
    .bind(post_id)
    .fetch_all(pool)
    .await?;
    Ok(calls)
}

pub async fn count_model_calls_by_post(pool: &SqlitePool, post_id: i64) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM model_call WHERE post_id = ?")
        .bind(post_id)
        .fetch_one(pool)
        .await?;
    Ok(count.0)
}

pub async fn count_whitelisted_posts(pool: &SqlitePool, feed_id: i64) -> AppResult<i64> {
    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM post WHERE feed_id = ? AND whitelisted = 1",
    )
    .bind(feed_id)
    .fetch_one(pool)
    .await?;
    Ok(count.0)
}

// ── ProcessingJob queries ──

pub async fn get_active_jobs(pool: &SqlitePool, limit: i64) -> AppResult<Vec<ProcessingJob>> {
    let jobs = sqlx::query_as::<_, ProcessingJob>(
        "SELECT * FROM processing_job WHERE status IN ('pending', 'running') ORDER BY created_at DESC LIMIT ?",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(jobs)
}

pub async fn get_all_jobs(pool: &SqlitePool, limit: i64) -> AppResult<Vec<ProcessingJob>> {
    let jobs = sqlx::query_as::<_, ProcessingJob>(
        "SELECT * FROM processing_job ORDER BY created_at DESC LIMIT ?",
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(jobs)
}

pub async fn get_job_by_id(pool: &SqlitePool, id: &str) -> AppResult<Option<ProcessingJob>> {
    let job = sqlx::query_as::<_, ProcessingJob>("SELECT * FROM processing_job WHERE id = ?")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    Ok(job)
}
