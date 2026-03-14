use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::middleware::{get_auth_user, require_admin_user};
use crate::auth::AuthenticatedUser;
use crate::db::queries;
use crate::error::{AppError, AppResult};
use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/feeds", get(list_feeds))
        .route("/feed", post(add_feed))
        .route("/feed/{feed_id}", get(serve_feed).delete(delete_feed))
        .route("/api/feeds/search", get(search_feeds))
        .route("/api/feeds/{feed_id}/refresh", post(refresh_feed))
        .route("/api/feeds/refresh-all", post(refresh_all_feeds))
        .route("/api/feeds/{feed_id}/settings", patch(update_feed_settings))
        .route("/api/feeds/{feed_id}/join", post(join_feed))
        .route("/api/feeds/{feed_id}/leave", post(leave_feed))
        .route("/api/feeds/{feed_id}/exit", post(exit_feed))
        .route("/api/feeds/{feed_id}/share-link", post(create_share_link))
        .route(
            "/api/feeds/{feed_id}/toggle-whitelist-all",
            post(toggle_whitelist_all),
        )
        .route("/api/feeds/aggregate", get(aggregate_feed))
        .route("/feed/user/{user_id}", get(user_feed))
        .route("/api/user/aggregate-link", post(create_aggregate_link))
}

async fn serialize_feed(
    pool: &sqlx::SqlitePool,
    feed: &crate::db::models::Feed,
    user_id: Option<i64>,
) -> Value {
    let posts_count: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM post WHERE feed_id = ?",
    )
    .bind(feed.id)
    .fetch_one(pool)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    let member_count: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_feed WHERE feed_id = ?",
    )
    .bind(feed.id)
    .fetch_one(pool)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    let is_member = if let Some(uid) = user_id {
        sqlx::query_as::<_, (i64,)>(
            "SELECT COUNT(*) FROM user_feed WHERE user_id = ? AND feed_id = ?",
        )
        .bind(uid)
        .bind(feed.id)
        .fetch_one(pool)
        .await
        .map(|(c,)| c > 0)
        .unwrap_or(false)
    } else {
        false
    };

    json!({
        "id": feed.id,
        "title": feed.title,
        "rss_url": feed.rss_url,
        "description": feed.description,
        "author": feed.author,
        "image_url": feed.image_url,
        "auto_whitelist_new_episodes_override": feed.auto_whitelist_new_episodes_override,
        "posts_count": posts_count,
        "member_count": member_count,
        "is_member": is_member,
        "is_active_subscription": is_member,
        "ad_detection_strategy": &feed.ad_detection_strategy,
        "chapter_filter_strings": feed.chapter_filter_strings,
    })
}

async fn list_feeds(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    let mut feeds = if state.config.require_auth {
        if let Some(user) = get_auth_user(&auth_user) {
            if user.role == "admin" {
                queries::get_all_feeds(&state.db).await?
            } else {
                let mut user_feeds = queries::get_user_visible_feeds(&state.db, user.id).await?;
                // Python parity: Feed #1 is always visible to all users
                if !user_feeds.iter().any(|f| f.id == 1) {
                    if let Ok(Some(feed_1)) = queries::get_feed_by_id(&state.db, 1).await {
                        user_feeds.push(feed_1);
                    }
                }
                user_feeds
            }
        } else {
            return Err(AppError::Unauthorized("Authentication required.".into()));
        }
    } else {
        queries::get_all_feeds(&state.db).await?
    };
    feeds.sort_by(|a, b| a.title.cmp(&b.title));

    let mut feeds_json = Vec::new();
    let user_id = get_auth_user(&auth_user).map(|u| u.id);
    for feed in &feeds {
        feeds_json.push(serialize_feed(&state.db, &feed, user_id).await);
    }

    // Python returns bare array, not wrapped object
    Ok(Json(json!(feeds_json)))
}

fn extract_feed_url(headers: &axum::http::HeaderMap, body: &[u8]) -> Result<String, AppError> {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if content_type.contains("application/json") {
        // JSON body: accept {"url": "..."} or {"rss_url": "..."}
        let parsed: Value = serde_json::from_slice(body)
            .map_err(|_| AppError::BadRequest("Invalid JSON".into()))?;
        let url = parsed
            .get("url")
            .or_else(|| parsed.get("rss_url"))
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| AppError::BadRequest("url is required.".into()))?;
        Ok(url)
    } else {
        // FormData: parse URL-encoded or multipart form with "url" field
        let text = std::str::from_utf8(body).unwrap_or("");
        // Try URL-encoded form first
        for pair in text.split('&') {
            let mut kv = pair.splitn(2, '=');
            if let (Some(key), Some(val)) = (kv.next(), kv.next()) {
                if key == "url" {
                    let decoded = urlencoding::decode(val)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|_| val.to_string());
                    let trimmed = decoded.trim().to_string();
                    if !trimmed.is_empty() {
                        return Ok(trimmed);
                    }
                }
            }
        }
        // Try multipart boundary parsing (simplified: look for name="url")
        if let Some(pos) = text.find("name=\"url\"") {
            let after = &text[pos + 10..];
            // Skip past \r\n\r\n
            if let Some(body_start) = after.find("\r\n\r\n") {
                let val_start = body_start + 4;
                let remaining = &after[val_start..];
                // Value ends at next boundary
                if let Some(end) = remaining.find("\r\n--") {
                    let val = remaining[..end].trim();
                    if !val.is_empty() {
                        return Ok(val.to_string());
                    }
                }
            }
        }
        Err(AppError::BadRequest("url is required.".into()))
    }
}

async fn add_feed(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<impl IntoResponse, AppError> {
    // Accept both FormData (frontend sends "url") and JSON (API sends "rss_url" or "url")
    let rss_url = extract_feed_url(&headers, &body)?;

    // Check if feed already exists
    if let Some(existing) = queries::get_feed_by_rss_url(&state.db, &rss_url).await? {
        if let Some(user) = get_auth_user(&auth_user) {
            if !queries::is_feed_member(&state.db, user.id, existing.id).await? {
                let allowance = user_feed_allowance(&state, user).await?;
                let current = queries::count_user_feeds(&state.db, user.id).await?;
                if current >= allowance {
                    return Err(AppError::PaymentRequired("Feed limit reached.".into()));
                }
            }
            queries::ensure_feed_membership(&state.db, user.id, existing.id).await?;
        }
        return Ok((StatusCode::OK, Json(json!({"feed": {"id": existing.id, "title": existing.title}}))));
    }

    // Parse RSS
    let parsed = crate::feeds::parser::fetch_and_parse(&rss_url)
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to parse RSS: {e}")))?;

    let feed_id = queries::insert_feed(
        &state.db,
        &parsed.title,
        &rss_url,
        parsed.description.as_deref(),
        parsed.author.as_deref(),
        parsed.image_url.as_deref(),
    )
    .await?;

    // Add membership
    if let Some(user) = get_auth_user(&auth_user) {
        queries::ensure_feed_membership(&state.db, user.id, feed_id).await?;
    }

    // Refresh feed to populate episodes
    let app = queries::get_app_settings(&state.db).await?;
    let _ = crate::feeds::refresh::refresh_feed(&state.db, feed_id, &rss_url).await;

    // Auto-whitelist latest episode for first member (Python parity)
    whitelist_latest_for_first_member(&state.db, feed_id).await;

    // Enqueue processing for whitelisted posts
    if app.autoprocess_on_download {
        state.jobs_manager.enqueue_pending_jobs().await;
    }

    let feed = queries::get_feed_by_id(&state.db, feed_id).await?;

    Ok((StatusCode::CREATED, Json(json!({"feed": feed}))))
}

async fn serve_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    Query(token_params): Query<FeedTokenQuery>,
) -> Result<Response, AppError> {
    let feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    // Refresh feed before serving (Python parity)
    let _ = crate::feeds::refresh::refresh_feed(&state.db, feed_id, &feed.rss_url).await;

    // Python: when autoprocess_on_download=true, include ALL posts (even non-whitelisted)
    // Otherwise only include whitelisted + processed posts
    let app = queries::get_app_settings(&state.db).await?;
    let posts = if app.autoprocess_on_download {
        queries::get_all_posts_for_feed(&state.db, feed_id).await?
    } else {
        queries::get_whitelisted_posts_for_feed(&state.db, feed_id).await?
    };

    let base = base_url(&state);
    let token_suffix = build_token_suffix(&token_params);

    let xml = crate::feeds::generator::generate_rss_feed(&feed, &posts, &base, &token_suffix)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("XML generation error: {e}")))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/rss+xml; charset=utf-8")
        .body(axum::body::Body::from(xml))
        .unwrap()
        .into_response())
}

async fn delete_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let _feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    let posts = queries::get_all_posts_for_feed(&state.db, feed_id).await?;
    for post in &posts {
        state.jobs_manager.cancel_post_jobs(&post.guid).await;
    }

    let audio_paths = queries::delete_feed_cascade(&state.db, feed_id).await?;

    for (processed, unprocessed) in &audio_paths {
        if let Some(p) = processed {
            let _ = tokio::fs::remove_file(p).await;
        }
        if let Some(p) = unprocessed {
            let _ = tokio::fs::remove_file(p).await;
        }
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

#[derive(Deserialize)]
struct SearchQuery {
    term: Option<String>,
}

async fn search_feeds(
    State(state): State<AppState>,
    Query(q): Query<SearchQuery>,
) -> AppResult<Json<Value>> {
    let query = q
        .term
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Search query is required.".into()))?;

    let (api_key, api_secret) = match (
        &state.config.podcast_index_api_key,
        &state.config.podcast_index_api_secret,
    ) {
        (Some(k), Some(s)) if !k.is_empty() && !s.is_empty() => (k.clone(), s.clone()),
        _ => return Ok(Json(json!({"results": []}))),
    };

    let epoch = chrono::Utc::now().timestamp();
    let auth_string = format!("{api_key}{api_secret}{epoch}");
    let hash = sha1_hex(&auth_string);

    let client = reqwest::Client::new();
    let resp = client
        .get("https://api.podcastindex.org/api/1.0/search/byterm")
        .header("X-Auth-Key", &api_key)
        .header("X-Auth-Date", epoch.to_string())
        .header("Authorization", &hash)
        .header("User-Agent", "PodlyPurePodcasts/1.0")
        .query(&[("q", query)])
        .send()
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Podcast Index API error: {e}")))?;

    let data: Value = resp
        .json()
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Invalid API response: {e}")))?;

    let results: Vec<Value> = data
        .get("feeds")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|f| {
                    json!({
                        "title": f.get("title").and_then(|v| v.as_str()),
                        "url": f.get("url").and_then(|v| v.as_str()),
                        "description": f.get("description").and_then(|v| v.as_str()),
                        "author": f.get("author").and_then(|v| v.as_str()),
                        "image": f.get("image").and_then(|v| v.as_str()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let total = results.len();
    Ok(Json(json!({"results": results, "total": total})))
}

fn sha1_hex(input: &str) -> String {
    use sha1::Digest;
    let mut hasher = sha1::Sha1::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

async fn refresh_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    let result = crate::feeds::refresh::refresh_feed(&state.db, feed_id, &feed.rss_url).await;

    match result {
        Ok(r) => {
            state.jobs_manager.enqueue_pending_jobs().await;
            Ok(Json(json!({
                "status": "ok",
                "new_episodes": r.new_episodes,
                "total_episodes": r.total_episodes,
            })))
        }
        Err(e) => Err(AppError::Internal(anyhow::anyhow!("Refresh failed: {e}"))),
    }
}

async fn refresh_all_feeds(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    state.jobs_manager.start_refresh_all_feeds().await;

    Ok(Json(json!({"status": "ok", "message": "Refresh started."})))
}

#[derive(Deserialize)]
struct UpdateFeedSettingsRequest {
    ad_detection_strategy: Option<String>,
    chapter_filter_strings: Option<String>,
    auto_whitelist_new_episodes_override: Option<Value>,
}

async fn update_feed_settings(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<UpdateFeedSettingsRequest>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let _feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    if let Some(strategy) = &body.ad_detection_strategy {
        queries::update_feed_strategy(&state.db, feed_id, strategy).await?;
    }
    if let Some(filter) = &body.chapter_filter_strings {
        let val = if filter.is_empty() { None } else { Some(filter.as_str()) };
        queries::update_feed_chapter_filter(&state.db, feed_id, val).await?;
    }
    if let Some(override_val) = &body.auto_whitelist_new_episodes_override {
        let val = if override_val.is_null() { None } else { override_val.as_bool() };
        queries::update_feed_auto_whitelist(&state.db, feed_id, val).await?;
    }

    // Re-fetch and return full feed object (Python parity)
    let updated_feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;
    let user_id = get_auth_user(&auth_user).map(|u| u.id);
    Ok(Json(serialize_feed(&state.db, &updated_feed, user_id).await))
}

async fn join_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;

    let feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    if queries::is_feed_member(&state.db, user.id, feed_id).await? {
        return Ok(Json(serialize_feed(&state.db, &feed, Some(user.id)).await));
    }

    let allowance = user_feed_allowance(&state, user).await?;
    let current = queries::count_user_feeds(&state.db, user.id).await?;
    if current >= allowance {
        return Err(AppError::PaymentRequired(
            serde_json::to_string(&json!({
                "error": "FEED_LIMIT_REACHED",
                "message": format!("Your plan allows {} feeds. Increase your plan to add more.", allowance),
                "feeds_in_use": current,
                "feed_allowance": allowance,
            }))
            .unwrap_or_else(|_| "Feed limit reached.".into()),
        ));
    }

    queries::ensure_feed_membership(&state.db, user.id, feed_id).await?;

    // Auto-whitelist latest episode for first member
    whitelist_latest_for_first_member(&state.db, feed_id).await;

    Ok(Json(serialize_feed(&state.db, &feed, Some(user.id)).await))
}

async fn leave_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    queries::remove_feed_membership(&state.db, user.id, feed_id).await?;
    let feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;
    Ok(Json(serialize_feed(&state.db, &feed, Some(user.id)).await))
}

/// Alias for leave_feed (Python has both /leave and /exit)
async fn exit_feed(
    state: State<AppState>,
    path: Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    leave_feed(state, path, auth_user).await
}

async fn create_share_link(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;

    let _feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    let (token_id, secret) =
        crate::auth::feed_tokens::create_feed_access_token(&state.db, user.id, Some(feed_id))
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Token error: {e}")))?;

    let url = format!(
        "{}/feed/{}?feed_token={}&feed_secret={}",
        base_url(&state),
        feed_id,
        token_id,
        secret
    );

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "url": url,
            "feed_token": token_id,
            "feed_secret": secret,
            "feed_id": feed_id,
        })),
    )
        .into_response())
}

#[derive(Deserialize)]
struct WhitelistAllRequest {
    whitelist: Option<bool>,
}

async fn toggle_whitelist_all(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<WhitelistAllRequest>,
) -> AppResult<Json<Value>> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let _feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    let whitelist = body.whitelist.unwrap_or(true);

    // Count total posts and how many changed
    let total: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM post WHERE feed_id = ?",
    )
    .bind(feed_id)
    .fetch_one(&state.db)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    let updated: u64 = queries::set_all_posts_whitelist(&state.db, feed_id, whitelist).await?;

    if whitelist {
        state.jobs_manager.enqueue_pending_jobs().await;
    }

    let whitelisted_count = if whitelist { total } else { 0 };

    Ok(Json(json!({
        "message": if whitelist { "All episodes whitelisted" } else { "All episodes unwhitelisted" },
        "whitelisted_count": whitelisted_count,
        "total_count": total,
        "all_whitelisted": whitelist,
        "updated_count": updated,
    })))
}

async fn aggregate_feed(
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    Ok(Json(json!({"redirect": format!("/feed/user/{}", user.id)})))
}

async fn user_feed(
    State(state): State<AppState>,
    Path(user_id): Path<i64>,
    Query(token_params): Query<FeedTokenQuery>,
) -> Result<Response, AppError> {
    let user = queries::get_user_by_id(&state.db, user_id)
        .await?
        .ok_or(AppError::NotFound)?;

    let user_feeds = queries::get_user_feeds(&state.db, user_id).await?;
    let mut all_posts = Vec::new();

    for uf in &user_feeds {
        let posts = queries::get_recent_processed_posts(&state.db, uf.feed_id, 3).await?;
        all_posts.extend(posts);
    }

    all_posts.sort_by(|a, b| b.release_date.cmp(&a.release_date));

    let (title, description) = if state.config.require_auth {
        (
            format!("Podly Podcasts - {}", user.username),
            format!(
                "Aggregate feed for {} - Last 3 processed episodes from each subscribed feed.",
                user.username
            ),
        )
    } else {
        (
            "Podly Podcasts".into(),
            "Aggregate feed - Last 3 processed episodes from each subscribed feed.".into(),
        )
    };

    let token_suffix = build_token_suffix(&token_params);
    let xml = crate::feeds::generator::generate_aggregate_rss_feed(
        &title,
        &description,
        user_id,
        &all_posts,
        &base_url(&state),
        &token_suffix,
    )
    .map_err(|e| AppError::Internal(anyhow::anyhow!("XML generation error: {e}")))?;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, "application/rss+xml; charset=utf-8")
        .body(axum::body::Body::from(xml))
        .unwrap()
        .into_response())
}

async fn create_aggregate_link(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;

    let (token_id, secret) =
        crate::auth::feed_tokens::create_feed_access_token(&state.db, user.id, None)
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Token error: {e}")))?;

    let url = format!(
        "{}/feed/user/{}?feed_token={}&feed_secret={}",
        base_url(&state),
        user.id,
        token_id,
        secret
    );

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "url": url,
            "feed_token": token_id,
            "feed_secret": secret,
        })),
    )
        .into_response())
}

#[derive(Deserialize)]
struct FeedTokenQuery {
    feed_token: Option<String>,
    feed_secret: Option<String>,
}

fn build_token_suffix(params: &FeedTokenQuery) -> String {
    match (&params.feed_token, &params.feed_secret) {
        (Some(token), Some(secret)) => {
            format!("?feed_token={}&feed_secret={}", token, secret)
        }
        _ => String::new(),
    }
}

fn base_url(state: &AppState) -> String {
    format!("http://{}:{}", state.config.host, state.config.port)
}

async fn user_feed_allowance(state: &AppState, user: &AuthenticatedUser) -> Result<i64, AppError> {
    let db_user = queries::get_user_by_id(&state.db, user.id)
        .await?
        .ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    Ok(db_user.manual_feed_allowance.unwrap_or(db_user.feed_allowance))
}

/// Whitelist the latest episode when the first member joins a feed (Python parity).
async fn whitelist_latest_for_first_member(pool: &sqlx::SqlitePool, feed_id: i64) {
    // Only whitelist if no posts are currently whitelisted
    let whitelisted_count: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM post WHERE feed_id = ? AND whitelisted = 1",
    )
    .bind(feed_id)
    .fetch_one(pool)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    if whitelisted_count > 0 {
        return;
    }

    // Whitelist the most recent episode
    let _ = sqlx::query(
        "UPDATE post SET whitelisted = 1 WHERE id = (SELECT id FROM post WHERE feed_id = ? ORDER BY release_date DESC LIMIT 1)",
    )
    .bind(feed_id)
    .execute(pool)
    .await;
}
