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
        .route("/feed/aggregate", get(aggregate_feed_legacy))
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
    user: Option<&AuthenticatedUser>,
    require_auth: bool,
) -> Value {
    let user_id = user.map(|u| u.id);
    let posts_count: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM post WHERE feed_id = ?",
    )
    .bind(feed.id)
    .fetch_one(pool)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    let member_count: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM feed_supporter WHERE feed_id = ?",
    )
    .bind(feed.id)
    .fetch_one(pool)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    // Python parity: when auth is disabled, is_member is always true
    // Also Feed 1 is always treated as member when user is present or auth disabled
    let is_member = if !require_auth {
        true
    } else if feed.id == 1 && user.is_some() {
        true
    } else if let Some(uid) = user_id {
        sqlx::query_as::<_, (i64,)>(
            "SELECT COUNT(*) FROM feed_supporter WHERE user_id = ? AND feed_id = ?",
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

    // Compute is_active_subscription: checks if feed is within user's allowance
    let is_active_subscription = if is_member {
        if let Some(u) = user {
            is_feed_active_for_user(pool, feed.id, u).await
        } else if !require_auth {
            true
        } else {
            false
        }
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
        "is_active_subscription": is_active_subscription,
        "ad_detection_strategy": &feed.ad_detection_strategy,
        "chapter_filter_strings": feed.chapter_filter_strings,
    })
}

/// Check if a feed is within the user's allowance based on subscription date.
async fn is_feed_active_for_user(
    pool: &sqlx::SqlitePool,
    feed_id: i64,
    user: &AuthenticatedUser,
) -> bool {
    if user.role == "admin" {
        return true;
    }

    // Hack: Always treat Feed 1 as active (matches Python)
    if feed_id == 1 {
        return true;
    }

    let allowance = user.manual_feed_allowance.unwrap_or(user.feed_allowance) as usize;

    // Get user's feeds sorted by creation date to determine priority
    let user_feeds: Vec<(i64,)> = sqlx::query_as(
        "SELECT feed_id FROM feed_supporter WHERE user_id = ? ORDER BY created_at ASC",
    )
    .bind(user.id)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    for (i, (fid,)) in user_feeds.iter().enumerate() {
        if *fid == feed_id {
            return i < allowance;
        }
    }

    false
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
    // Python does not sort feeds — return in database order
    // feeds.sort_by(|a, b| a.title.cmp(&b.title));

    let mut feeds_json = Vec::new();
    let user_ref = get_auth_user(&auth_user);
    for feed in &feeds {
        feeds_json.push(serialize_feed(&state.db, &feed, user_ref, state.config.require_auth).await);
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
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    // Update last_active if user is present (Python parity)
    if let Some(user) = get_auth_user(&auth_user) {
        let _ = queries::update_user_last_active(&state.db, user.id).await;
    }

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
                        "feedUrl": f.get("url").and_then(|v| v.as_str()),
                        "description": f.get("description").and_then(|v| v.as_str()),
                        "author": f.get("author").and_then(|v| v.as_str()),
                        "artworkUrl": f.get("image").and_then(|v| v.as_str()),
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
    _auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<impl IntoResponse> {
    // Python does not require admin for refresh — any authenticated user can refresh

    let feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;

    let feed_title = feed.title.clone();

    // Spawn background refresh (matches Python's Thread-based approach)
    let db = state.db.clone();
    let rss_url = feed.rss_url.clone();
    let jobs = state.jobs_manager.clone();
    tokio::spawn(async move {
        if let Ok(_r) = crate::feeds::refresh::refresh_feed(&db, feed_id, &rss_url).await {
            jobs.enqueue_pending_jobs().await;
        }
    });

    Ok((StatusCode::ACCEPTED, Json(json!({
        "status": "accepted",
        "message": format!("Feed \"{}\" refresh queued for processing", feed_title),
    }))))
}

async fn refresh_all_feeds(
    State(state): State<AppState>,
    _auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    // Python does not require admin for refresh-all

    let result = state.jobs_manager.start_refresh_all_feeds().await;

    Ok(Json(json!({
        "status": "success",
        "feeds_refreshed": result.get("feeds_refreshed"),
        "jobs_enqueued": result.get("enqueued"),
    })))
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
        const VALID_STRATEGIES: &[&str] = &["inherit", "llm", "oneshot", "chapter"];
        if !VALID_STRATEGIES.contains(&strategy.as_str()) {
            return Err(AppError::BadRequest(format!(
                "Invalid ad_detection_strategy: {}. Must be one of: {}",
                strategy,
                VALID_STRATEGIES.join(", ")
            )));
        }
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
    let user_ref = get_auth_user(&auth_user);
    Ok(Json(serialize_feed(&state.db, &updated_feed, user_ref, state.config.require_auth).await))
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
        return Ok(Json(serialize_feed(&state.db, &feed, Some(user), state.config.require_auth).await));
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

    Ok(Json(serialize_feed(&state.db, &feed, Some(user), state.config.require_auth).await))
}

async fn leave_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    queries::remove_feed_membership(&state.db, user.id, feed_id).await?;

    // Python returns {status: "ok", feed_id}
    Ok(Json(json!({
        "status": "ok",
        "feed_id": feed_id,
    })))
}

/// Python /exit returns the full serialized feed (different from /leave)
async fn exit_feed(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> AppResult<Json<Value>> {
    let user = get_auth_user(&auth_user).ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    queries::remove_feed_membership(&state.db, user.id, feed_id).await?;
    let feed = queries::get_feed_by_id(&state.db, feed_id)
        .await?
        .ok_or(AppError::NotFound)?;
    Ok(Json(serialize_feed(&state.db, &feed, Some(user), state.config.require_auth).await))
}

async fn create_share_link(
    State(state): State<AppState>,
    Path(feed_id): Path<i64>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    // Python returns 404 when auth is disabled
    if !state.config.require_auth {
        return Err(AppError::NotFoundMsg("Authentication is disabled.".into()));
    }
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

    // Count total posts
    let total: i64 = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM post WHERE feed_id = ?",
    )
    .bind(feed_id)
    .fetch_one(&state.db)
    .await
    .map(|(c,)| c)
    .unwrap_or(0);

    // Python parity: auto-toggle based on current state
    // If all are whitelisted → unwhitelist all; otherwise → whitelist all
    let whitelist = match body.whitelist {
        Some(v) => v,
        None => {
            let whitelisted_count: i64 = sqlx::query_as::<_, (i64,)>(
                "SELECT COUNT(*) FROM post WHERE feed_id = ? AND whitelisted = 1",
            )
            .bind(feed_id)
            .fetch_one(&state.db)
            .await
            .map(|(c,)| c)
            .unwrap_or(0);
            // Toggle: if not all whitelisted, whitelist all; otherwise unwhitelist all
            whitelisted_count < total
        }
    };

    let updated: u64 = queries::set_all_posts_whitelist(&state.db, feed_id, whitelist).await?;

    if whitelist {
        state.jobs_manager.enqueue_pending_jobs().await;
    }

    let whitelisted_count = if whitelist { total } else { 0 };

    Ok(Json(json!({
        "message": if whitelist { "Whitelisted all posts" } else { "Unwhitelisted all posts" },
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

/// Legacy /feed/aggregate route — Python serves this as a redirect to the user's aggregate feed.
async fn aggregate_feed_legacy(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Query(token_params): Query<FeedTokenQuery>,
) -> Result<Response, AppError> {
    let user_id = if let Some(user) = get_auth_user(&auth_user) {
        user.id
    } else if !state.config.require_auth {
        // When auth disabled, find the admin user (or fallback to user_id=0)
        let admin_id: Option<(i64,)> = sqlx::query_as(
            "SELECT id FROM user WHERE role = 'admin' ORDER BY id ASC LIMIT 1",
        )
        .fetch_optional(&state.db)
        .await
        .ok()
        .flatten();
        admin_id.map(|(id,)| id).unwrap_or(0)
    } else {
        return Err(AppError::Unauthorized("Authentication required.".into()));
    };
    let token_suffix = build_token_suffix(&token_params);
    let redirect_url = format!("/feed/user/{}{}", user_id, token_suffix);
    Ok(axum::response::Redirect::to(&redirect_url).into_response())
}

async fn user_feed(
    State(state): State<AppState>,
    Path(user_id): Path<i64>,
    Query(token_params): Query<FeedTokenQuery>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Response, AppError> {
    // Python parity: auth check — only admin or the user themselves can access
    if state.config.require_auth {
        let current = get_auth_user(&auth_user)
            .ok_or(AppError::Unauthorized("Authentication required".into()))?;
        if current.role != "admin" && current.id != user_id {
            return Err(AppError::ForbiddenMsg("Forbidden".into()));
        }
    }

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
    let user_id = if let Some(user) = get_auth_user(&auth_user) {
        user.id
    } else if !state.config.require_auth {
        // When auth disabled, find or create default admin user
        let admin_id: Option<(i64,)> = sqlx::query_as(
            "SELECT id FROM user WHERE role = 'admin' ORDER BY id ASC LIMIT 1",
        )
        .fetch_optional(&state.db)
        .await
        .ok()
        .flatten();
        admin_id.map(|(id,)| id).unwrap_or(1)
    } else {
        return Err(AppError::Unauthorized("Authentication required.".into()));
    };

    let (token_id, secret) =
        crate::auth::feed_tokens::create_feed_access_token(&state.db, user_id, None)
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Token error: {e}")))?;

    // When auth disabled, Python doesn't include token params in URL
    let url = if state.config.require_auth {
        format!(
            "{}/feed/user/{}?feed_token={}&feed_secret={}",
            base_url(&state), user_id, token_id, secret
        )
    } else {
        format!("{}/feed/user/{}", base_url(&state), user_id)
    };

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
    if let Some(url) = &state.config.base_url {
        return url.clone();
    }
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
