use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tower_sessions::Session;

use crate::auth::middleware::require_admin_user;
use crate::auth::AuthenticatedUser;
use crate::db::queries;
use crate::error::{AppError, AppResult};
use crate::AppState;

const SESSION_USER_KEY: &str = "user_id";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/auth/status", get(auth_status))
        .route("/api/auth/login", post(login))
        .route("/api/auth/logout", post(logout))
        .route("/api/auth/me", get(me))
        .route("/api/auth/change-password", post(change_password))
        .route("/api/auth/users", get(list_users).post(create_user))
        .route(
            "/api/auth/users/{username}",
            axum::routing::patch(update_user).delete(delete_user),
        )
        .route("/api/auth/discord/status", get(discord_status))
}

async fn auth_status(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let app_settings = queries::get_app_settings(&state.db).await?;
    Ok(Json(json!({
        "require_auth": state.config.require_auth,
        "landing_page_enabled": app_settings.enable_public_landing_page,
    })))
}

#[derive(Deserialize)]
struct LoginRequest {
    username: Option<String>,
    password: Option<String>,
}

async fn login(
    State(state): State<AppState>,
    session: Session,
    Json(body): Json<LoginRequest>,
) -> Result<impl IntoResponse, AppError> {
    if !state.config.require_auth {
        return Err(AppError::NotFound);
    }

    let username = body
        .username
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Username and password are required.".into()))?;
    let password = body
        .password
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Username and password are required.".into()))?;

    // Rate limiting
    let client_ip = "unknown".to_string();
    {
        let mut limiter = state.rate_limiter.lock().await;
        if let Some(retry_after) = limiter.retry_after(&client_ip) {
            return Err(AppError::TooManyRequests { retry_after });
        }
    }

    let user = queries::get_user_by_username(&state.db, username).await?;
    let user = match user {
        Some(u) => {
            let valid =
                crate::auth::verify_password(password, &u.password_hash).unwrap_or(false);
            if !valid {
                let backoff = state.rate_limiter.lock().await.register_failure(&client_ip);
                if backoff > 0 {
                    return Err(AppError::TooManyRequests {
                        retry_after: backoff,
                    });
                }
                return Err(AppError::BadRequest("Invalid username or password.".into()));
            }
            u
        }
        None => {
            state.rate_limiter.lock().await.register_failure(&client_ip);
            return Err(AppError::BadRequest("Invalid username or password.".into()));
        }
    };

    state.rate_limiter.lock().await.register_success(&client_ip);
    session.clear().await;
    session
        .insert(SESSION_USER_KEY, user.id)
        .await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Session error: {e}")))?;

    let _ = queries::update_user_last_active(&state.db, user.id).await;

    let allowance = user.manual_feed_allowance.unwrap_or(user.feed_allowance);
    Ok(Json(json!({
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "feed_allowance": allowance,
            "feed_subscription_status": user.feed_subscription_status,
        }
    })))
}

async fn logout(
    State(state): State<AppState>,
    session: Session,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<impl IntoResponse, AppError> {
    if !state.config.require_auth {
        return Err(AppError::NotFound);
    }
    if auth_user.is_none() {
        session.clear().await;
        return Err(AppError::Unauthorized);
    }
    session.clear().await;
    Ok(StatusCode::NO_CONTENT)
}

async fn me(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Json<Value>, AppError> {
    if !state.config.require_auth {
        return Err(AppError::NotFound);
    }
    let Extension(auth_user) = auth_user.ok_or(AppError::Unauthorized)?;
    let user = queries::get_user_by_id(&state.db, auth_user.id)
        .await?
        .ok_or(AppError::Unauthorized)?;
    let allowance = user.manual_feed_allowance.unwrap_or(user.feed_allowance);
    Ok(Json(json!({
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
            "feed_allowance": allowance,
            "feed_subscription_status": user.feed_subscription_status,
        }
    })))
}

#[derive(Deserialize)]
struct ChangePasswordRequest {
    current_password: Option<String>,
    new_password: Option<String>,
}

async fn change_password(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<ChangePasswordRequest>,
) -> Result<Json<Value>, AppError> {
    if !state.config.require_auth {
        return Err(AppError::NotFound);
    }
    let Extension(auth_user) = auth_user.ok_or(AppError::Unauthorized)?;
    let user = queries::get_user_by_id(&state.db, auth_user.id)
        .await?
        .ok_or(AppError::Unauthorized)?;

    let current = body.current_password.as_deref().filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Current and new passwords are required.".into()))?;
    let new_pass = body.new_password.as_deref().filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Current and new passwords are required.".into()))?;

    if !crate::auth::verify_password(current, &user.password_hash).unwrap_or(false) {
        return Err(AppError::Unauthorized);
    }
    crate::auth::validate_password(new_pass).map_err(AppError::BadRequest)?;

    let new_hash = crate::auth::hash_password(new_pass)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Hash error: {e}")))?;
    queries::update_user_password(&state.db, user.id, &new_hash).await?;
    Ok(Json(json!({"status": "ok"})))
}

async fn list_users(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Json<Value>, AppError> {
    if !state.config.require_auth {
        return Err(AppError::NotFound);
    }
    require_admin_user(&auth_user, state.config.require_auth)?;

    let users = queries::get_all_users(&state.db).await?;
    let users_json: Vec<Value> = users.iter().map(|u| json!({
        "id": u.id, "username": u.username, "role": u.role,
        "created_at": u.created_at, "updated_at": u.updated_at,
        "last_active": u.last_active, "feed_allowance": u.feed_allowance,
        "manual_feed_allowance": u.manual_feed_allowance,
        "feed_subscription_status": u.feed_subscription_status,
    })).collect();
    Ok(Json(json!({"users": users_json})))
}

#[derive(Deserialize)]
struct CreateUserRequest {
    username: Option<String>,
    password: Option<String>,
    role: Option<String>,
}

async fn create_user(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<CreateUserRequest>,
) -> Result<impl IntoResponse, AppError> {
    if !state.config.require_auth {
        return Err(AppError::NotFound);
    }
    require_admin_user(&auth_user, state.config.require_auth)?;

    let username = body.username.as_deref().map(|s| s.trim()).filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Username and password are required.".into()))?;
    let password = body.password.as_deref().filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("Username and password are required.".into()))?;
    let role = body.role.as_deref().unwrap_or("user");

    crate::auth::validate_password(password).map_err(AppError::BadRequest)?;

    if queries::get_user_by_username(&state.db, username).await?.is_some() {
        return Err(AppError::Conflict(format!("User '{username}' already exists.")));
    }

    let app_settings = queries::get_app_settings(&state.db).await?;
    if let Some(limit) = app_settings.user_limit_total {
        let count = queries::count_users(&state.db).await?;
        if count >= limit {
            return Err(AppError::BadRequest(format!("User limit of {limit} reached.")));
        }
    }

    let hash = crate::auth::hash_password(password)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Hash error: {e}")))?;
    let user_id = queries::insert_user(&state.db, username, &hash, role).await?;
    let user = queries::get_user_by_id(&state.db, user_id).await?
        .ok_or(AppError::Internal(anyhow::anyhow!("User not found after insert")))?;

    Ok((StatusCode::CREATED, Json(json!({
        "user": {
            "id": user.id, "username": user.username, "role": user.role,
            "created_at": user.created_at, "updated_at": user.updated_at,
        }
    }))))
}

#[derive(Deserialize)]
struct UpdateUserRequest {
    role: Option<String>,
    password: Option<String>,
    manual_feed_allowance: Option<Value>,
}

async fn update_user(
    State(state): State<AppState>,
    axum::extract::Path(username): axum::extract::Path<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<UpdateUserRequest>,
) -> Result<Json<Value>, AppError> {
    if !state.config.require_auth { return Err(AppError::NotFound); }
    require_admin_user(&auth_user, state.config.require_auth)?;

    let target = queries::get_user_by_username(&state.db, &username).await?.ok_or(AppError::NotFound)?;

    if let Some(role) = &body.role {
        if target.role == "admin" && role != "admin" {
            let admin_count = queries::count_admin_users(&state.db).await?;
            if admin_count <= 1 {
                return Err(AppError::BadRequest("Cannot remove the last admin.".into()));
            }
        }
        queries::update_user_role(&state.db, target.id, role).await?;
    }
    if let Some(password) = &body.password {
        if !password.is_empty() {
            crate::auth::validate_password(password).map_err(AppError::BadRequest)?;
            let hash = crate::auth::hash_password(password)
                .map_err(|e| AppError::Internal(anyhow::anyhow!("Hash error: {e}")))?;
            queries::update_user_password(&state.db, target.id, &hash).await?;
        }
    }
    if let Some(allowance) = &body.manual_feed_allowance {
        let val = if allowance.is_null() { None } else { allowance.as_i64() };
        queries::update_user_manual_feed_allowance(&state.db, target.id, val).await?;
    }
    Ok(Json(json!({"status": "ok"})))
}

async fn delete_user(
    State(state): State<AppState>,
    axum::extract::Path(username): axum::extract::Path<String>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Json<Value>, AppError> {
    if !state.config.require_auth { return Err(AppError::NotFound); }
    require_admin_user(&auth_user, state.config.require_auth)?;

    let target = queries::get_user_by_username(&state.db, &username).await?.ok_or(AppError::NotFound)?;
    if target.role == "admin" {
        let admin_count = queries::count_admin_users(&state.db).await?;
        if admin_count <= 1 {
            return Err(AppError::BadRequest("Cannot delete the last admin.".into()));
        }
    }
    queries::delete_user(&state.db, target.id).await?;
    Ok(Json(json!({"status": "ok"})))
}

async fn discord_status(State(state): State<AppState>) -> AppResult<Json<Value>> {
    let db_settings = queries::get_discord_settings(&state.db).await?;
    let client_id = state.config.discord_client_id.as_deref()
        .or(db_settings.client_id.as_deref())
        .unwrap_or("");
    let client_secret = state.config.discord_client_secret.as_deref()
        .or(db_settings.client_secret.as_deref())
        .unwrap_or("");
    let redirect_uri = state.config.discord_redirect_uri.as_deref()
        .or(db_settings.redirect_uri.as_deref())
        .unwrap_or("");
    let enabled = !client_id.is_empty() && !client_secret.is_empty() && !redirect_uri.is_empty();
    Ok(Json(json!({ "enabled": enabled })))
}
