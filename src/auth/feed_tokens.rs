use sha2::{Digest, Sha256};
use sqlx::SqlitePool;

use super::AuthenticatedUser;
use crate::db::models::FeedAccessToken;

#[allow(dead_code)]
pub struct FeedTokenAuthResult {
    pub user: AuthenticatedUser,
    pub feed_id: Option<i64>,
    pub token: FeedAccessToken,
}

pub fn hash_token(secret: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    hex::encode(hasher.finalize())
}

/// Create a new feed access token. Returns (token_id, secret).
pub async fn create_feed_access_token(
    pool: &SqlitePool,
    user_id: i64,
    feed_id: Option<i64>,
) -> Result<(String, String), sqlx::Error> {
    let token_id: String = uuid::Uuid::new_v4()
        .to_string()
        .replace('-', "")
        .chars()
        .take(32)
        .collect();
    let secret = uuid::Uuid::new_v4().to_string();
    let token_hash = hash_token(&secret);
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        "INSERT INTO feed_access_token (token_id, token_hash, token_secret, feed_id, user_id, created_at)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&token_id)
    .bind(&token_hash)
    .bind(&secret)
    .bind(feed_id)
    .bind(user_id)
    .bind(&now)
    .execute(pool)
    .await?;

    Ok((token_id, secret))
}

/// Authenticate a feed token from query params. Returns None if invalid.
pub async fn authenticate_feed_token(
    pool: &SqlitePool,
    token_id: &str,
    secret: &str,
    path: &str,
) -> Option<FeedTokenAuthResult> {
    let token = sqlx::query_as::<_, FeedAccessToken>(
        "SELECT * FROM feed_access_token WHERE token_id = ? AND revoked = 0",
    )
    .bind(token_id)
    .fetch_optional(pool)
    .await
    .ok()??;

    let expected_hash = hash_token(secret);
    if !constant_time_eq(token.token_hash.as_bytes(), expected_hash.as_bytes()) {
        return None;
    }

    let user = sqlx::query_as::<_, crate::db::models::User>("SELECT * FROM users WHERE id = ?")
        .bind(token.user_id)
        .fetch_optional(pool)
        .await
        .ok()??;

    if !validate_token_access(pool, &token, &user, path).await {
        return None;
    }

    // Touch last_used_at (fire and forget)
    let now = chrono::Utc::now().to_rfc3339();
    let _ = sqlx::query("UPDATE feed_access_token SET last_used_at = ? WHERE token_id = ?")
        .bind(&now)
        .bind(token_id)
        .execute(pool)
        .await;

    Some(FeedTokenAuthResult {
        user: AuthenticatedUser {
            id: user.id,
            username: user.username,
            role: user.role,
            feed_allowance: user.feed_allowance,
            manual_feed_allowance: user.manual_feed_allowance,
        },
        feed_id: token.feed_id,
        token,
    })
}

async fn validate_token_access(
    pool: &SqlitePool,
    token: &FeedAccessToken,
    user: &crate::db::models::User,
    path: &str,
) -> bool {
    if token.feed_id.is_none() {
        // Aggregate token
        if let Some(requested_uid) = resolve_user_id_from_feed_path(path) {
            return requested_uid == user.id;
        }
        if let Some(resource_feed_id) = resolve_feed_id(pool, path).await {
            return verify_subscription(pool, user, resource_feed_id).await;
        }
        return true;
    }

    // Specific feed token
    let token_feed_id = token.feed_id.unwrap();
    if let Some(feed_id) = resolve_feed_id(pool, path).await {
        if feed_id != token_feed_id {
            return false;
        }
        return verify_subscription(pool, user, token_feed_id).await;
    }

    false
}

async fn verify_subscription(
    pool: &SqlitePool,
    user: &crate::db::models::User,
    feed_id: i64,
) -> bool {
    if user.role == "admin" {
        return true;
    }
    // Hack: always allow feed 1 (matches Python behavior)
    if feed_id == 1 {
        return true;
    }

    let membership: Option<(i64,)> =
        sqlx::query_as("SELECT id FROM feed_supporter WHERE user_id = ? AND feed_id = ?")
            .bind(user.id)
            .bind(feed_id)
            .fetch_optional(pool)
            .await
            .unwrap_or(None);

    membership.is_some()
}

fn resolve_user_id_from_feed_path(path: &str) -> Option<i64> {
    let remainder = path.strip_prefix("/feed/user/")?;
    remainder.split('/').next()?.parse().ok()
}

async fn resolve_feed_id(pool: &SqlitePool, path: &str) -> Option<i64> {
    if let Some(remainder) = path.strip_prefix("/feed/") {
        return remainder.split('/').next()?.parse().ok();
    }

    if let Some(remainder) = path.strip_prefix("/api/posts/") {
        let guid = remainder.split('/').next()?;
        let post: Option<(i64,)> =
            sqlx::query_as("SELECT feed_id FROM post WHERE guid = ?")
                .bind(guid)
                .fetch_optional(pool)
                .await
                .ok()?;
        return post.map(|p| p.0);
    }

    if let Some(remainder) = path.strip_prefix("/post/") {
        let guid_part = remainder.split('/').next()?;
        let guid = guid_part.split('.').next()?;
        let post: Option<(i64,)> =
            sqlx::query_as("SELECT feed_id FROM post WHERE guid = ?")
                .bind(guid)
                .fetch_optional(pool)
                .await
                .ok()?;
        return post.map(|p| p.0);
    }

    None
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}
