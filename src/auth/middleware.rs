use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use tokio::sync::Mutex;

use super::rate_limiter::FailureRateLimiter;
use super::AuthenticatedUser;
use crate::AppState;

/// Public paths that never require auth.
const PUBLIC_PATHS: &[&str] = &[
    "/",
    "/health",
    "/robots.txt",
    "/manifest.json",
    "/favicon.ico",
    "/api/auth/login",
    "/api/auth/status",
    "/api/auth/discord/status",
    "/api/auth/discord/login",
    "/api/auth/discord/callback",
    "/api/landing/status",
    "/api/billing/stripe-webhook",
];

const PUBLIC_PREFIXES: &[&str] = &["/static/", "/assets/", "/images/", "/fonts/", "/.well-known/"];

const PUBLIC_EXTENSIONS: &[&str] = &[
    ".js", ".css", ".map", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".txt",
];

fn is_public_request(path: &str) -> bool {
    if PUBLIC_PATHS.contains(&path) {
        return true;
    }
    if PUBLIC_PREFIXES.iter().any(|p| path.starts_with(p)) {
        return true;
    }
    if PUBLIC_EXTENSIONS.iter().any(|e| path.ends_with(e)) {
        return true;
    }
    false
}

fn is_token_protected(path: &str) -> bool {
    // Match Python's exact regex patterns for token-protected paths
    let path_segments: Vec<&str> = path.split('/').collect();

    // ^/feed/[^/]+$ — /feed/<id> (exactly 2 segments after root)
    if path_segments.len() == 3 && path_segments[1] == "feed" && !path_segments[2].is_empty() {
        return true;
    }
    // ^/feed/user/[^/]+$ — /feed/user/<id> (exactly 3 segments after root)
    if path_segments.len() == 4 && path_segments[1] == "feed" && path_segments[2] == "user" && !path_segments[3].is_empty() {
        return true;
    }
    // ^/api/posts/[^/]+/(audio|download(?:/original)?)$
    if path_segments.len() >= 4 && path_segments[1] == "api" && path_segments[2] == "posts" {
        let action = path_segments[3..].join("/");
        // Must be <guid>/audio, <guid>/download, or <guid>/download/original
        if path_segments.len() == 5 && (path_segments[4] == "audio" || path_segments[4] == "download") {
            return true;
        }
        if path_segments.len() == 6 && path_segments[4] == "download" && path_segments[5] == "original" {
            return true;
        }
        let _ = action; // suppress unused warning
    }
    // ^/post/[^/]+(?:\.mp3|/original\.mp3)$
    if path_segments.len() >= 3 && path_segments[1] == "post" {
        let last = path_segments.last().unwrap_or(&"");
        if path_segments.len() == 3 && last.ends_with(".mp3") {
            return true;
        }
        if path_segments.len() == 4 && path_segments[3] == "original.mp3" {
            return true;
        }
    }
    false
}

/// Auth middleware that enforces authentication when `require_auth` is true.
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Response {
    if req.method() == axum::http::Method::OPTIONS {
        return next.run(req).await;
    }

    if !state.config.require_auth {
        return next.run(req).await;
    }

    let path = req.uri().path().to_string();

    if is_public_request(&path) {
        return next.run(req).await;
    }

    // Try session-based auth
    if let Some(session) = req.extensions().get::<tower_sessions::Session>() {
        if let Ok(Some(user_id)) = session.get::<i64>("user_id").await {
            if let Ok(Some(user)) = crate::db::queries::get_user_by_id(&state.db, user_id).await {
                let auth_user = AuthenticatedUser {
                    id: user.id,
                    username: user.username.clone(),
                    role: user.role.clone(),
                    feed_allowance: user.feed_allowance,
                    manual_feed_allowance: user.manual_feed_allowance,
                };
                req.extensions_mut().insert(auth_user);
                return next.run(req).await;
            }
        }
    }

    // For token-protected endpoints, try feed token auth
    if is_token_protected(&path) {
        let query = req.uri().query().unwrap_or("");
        let params: Vec<(String, String)> =
            form_urlencoded::parse(query.as_bytes())
                .map(|(k, v)| (k.into_owned(), v.into_owned()))
                .collect();

        let token_id = params
            .iter()
            .find(|(k, _)| k == "feed_token")
            .map(|(_, v)| v.as_str());
        let secret = params
            .iter()
            .find(|(k, _)| k == "feed_secret")
            .map(|(_, v)| v.as_str());

        if let (Some(tid), Some(sec)) = (token_id, secret) {
            let client_ip = req
                .headers()
                .get("x-forwarded-for")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown")
                .to_string();

            // Check rate limit
            {
                let mut limiter = state.rate_limiter.lock().await;
                if let Some(retry_after) = limiter.retry_after(&client_ip) {
                    return (
                        StatusCode::TOO_MANY_REQUESTS,
                        [("Retry-After", retry_after.to_string())],
                        "Too Many Authentication Attempts",
                    )
                        .into_response();
                }
            }

            if let Some(result) =
                super::feed_tokens::authenticate_feed_token(&state.db, tid, sec, &path).await
            {
                state
                    .rate_limiter
                    .lock()
                    .await
                    .register_success(&client_ip);
                req.extensions_mut().insert(result.user);
                return next.run(req).await;
            } else {
                let backoff = state
                    .rate_limiter
                    .lock()
                    .await
                    .register_failure(&client_ip);
                let mut resp =
                    (StatusCode::UNAUTHORIZED, "Invalid or missing feed token").into_response();
                if backoff > 0 {
                    resp.headers_mut().insert(
                        "Retry-After",
                        axum::http::HeaderValue::from_str(&backoff.to_string()).unwrap(),
                    );
                }
                return resp;
            }
        }
    }

    // No auth found
    (
        StatusCode::UNAUTHORIZED,
        axum::Json(json!({"error": "Authentication required."})),
    )
        .into_response()
}

/// Extractor helper: get the current user from request extensions.
/// Returns None if no user is authenticated.
#[allow(dead_code)]
pub fn get_current_user(extensions: &axum::http::Extensions) -> Option<AuthenticatedUser> {
    extensions.get::<AuthenticatedUser>().cloned()
}

/// Require admin user. Returns Some(user) if admin, None if auth disabled.
/// Returns AppError::Unauthorized or AppError::Forbidden on failure.
#[allow(dead_code)]
pub fn require_admin(
    extensions: &axum::http::Extensions,
    require_auth: bool,
) -> Result<Option<AuthenticatedUser>, crate::error::AppError> {
    if !require_auth {
        return Ok(None);
    }

    let user = extensions
        .get::<AuthenticatedUser>()
        .cloned()
        .ok_or(crate::error::AppError::Unauthorized("Authentication required.".into()))?;

    if user.role != "admin" {
        return Err(crate::error::AppError::Forbidden);
    }

    Ok(Some(user))
}

/// Convenience helper that checks admin from an Option<Extension<AuthenticatedUser>>.
/// Use in handlers that extract `auth_user: Option<Extension<AuthenticatedUser>>`.
pub fn require_admin_user(
    auth_user: &Option<axum::Extension<AuthenticatedUser>>,
    require_auth: bool,
) -> Result<(), crate::error::AppError> {
    if !require_auth {
        return Ok(());
    }
    let axum::Extension(user) = auth_user.as_ref().ok_or(crate::error::AppError::Unauthorized("Authentication required.".into()))?;
    if user.role != "admin" {
        return Err(crate::error::AppError::Forbidden);
    }
    Ok(())
}

/// Get authenticated user from Option<Extension<AuthenticatedUser>>.
pub fn get_auth_user(
    auth_user: &Option<axum::Extension<AuthenticatedUser>>,
) -> Option<&AuthenticatedUser> {
    auth_user.as_ref().map(|axum::Extension(u)| u)
}

// Re-export for AppState
pub type SharedRateLimiter = Arc<Mutex<FailureRateLimiter>>;
