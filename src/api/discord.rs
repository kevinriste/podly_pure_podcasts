use axum::extract::{Query, State};
use axum::response::{IntoResponse, Json, Redirect};
use axum::routing::get;
use axum::{Extension, Router};
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
        .route("/api/auth/discord/login", get(discord_login))
        .route("/api/auth/discord/callback", get(discord_callback))
        .route("/api/auth/discord/config", get(discord_config).put(discord_config_update))
}

/// Resolved Discord settings from DB + env overrides.
struct ResolvedDiscordSettings {
    enabled: bool,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    guild_ids: Vec<String>,
    allow_registration: bool,
}

async fn load_discord_settings(state: &AppState) -> AppResult<ResolvedDiscordSettings> {
    let db_settings = queries::get_discord_settings(&state.db).await?;

    // Env vars take precedence over DB
    let client_id = state.config.discord_client_id.clone()
        .or(db_settings.client_id)
        .unwrap_or_default();
    let client_secret = state.config.discord_client_secret.clone()
        .or(db_settings.client_secret)
        .unwrap_or_default();
    let redirect_uri = state.config.discord_redirect_uri.clone()
        .or(db_settings.redirect_uri)
        .unwrap_or_default();

    let guild_ids_str = state.config.discord_guild_ids.clone()
        .or(db_settings.guild_ids)
        .unwrap_or_default();
    let guild_ids: Vec<String> = guild_ids_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let allow_registration = state.config.discord_allow_registration
        .unwrap_or(db_settings.allow_registration);

    let enabled = !client_id.is_empty() && !client_secret.is_empty() && !redirect_uri.is_empty();

    Ok(ResolvedDiscordSettings {
        enabled,
        client_id,
        client_secret,
        redirect_uri,
        guild_ids,
        allow_registration,
    })
}

/// GET /api/auth/discord/login — returns OAuth authorization URL.
async fn discord_login(
    State(state): State<AppState>,
    session: Session,
) -> AppResult<Json<Value>> {
    let settings = load_discord_settings(&state).await?;
    if !settings.enabled {
        return Err(AppError::BadRequest("Discord OAuth is not configured.".into()));
    }

    // Generate CSRF state token
    let csrf_state = uuid::Uuid::new_v4().to_string();
    session.insert("discord_oauth_state", &csrf_state).await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Session error: {e}")))?;

    // Build scopes
    let mut scopes = vec!["identify"];
    if !settings.guild_ids.is_empty() {
        scopes.push("guilds");
    }

    let auth_url = format!(
        "https://discord.com/oauth2/authorize?client_id={}&redirect_uri={}&response_type=code&scope={}&state={}&prompt=none",
        urlencoded(&settings.client_id),
        urlencoded(&settings.redirect_uri),
        urlencoded(&scopes.join(" ")),
        urlencoded(&csrf_state),
    );

    Ok(Json(json!({ "url": auth_url })))
}

#[derive(Deserialize)]
struct CallbackQuery {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
}

/// GET /api/auth/discord/callback — handle Discord OAuth redirect.
async fn discord_callback(
    State(state): State<AppState>,
    session: Session,
    Query(params): Query<CallbackQuery>,
) -> Result<impl IntoResponse, AppError> {
    let settings = load_discord_settings(&state).await?;
    if !settings.enabled {
        return Ok(Redirect::to("/?error=discord_not_configured"));
    }

    // Handle Discord error responses
    if let Some(err) = &params.error {
        match err.as_str() {
            "access_denied" => return Ok(Redirect::to("/?error=access_denied")),
            _ => return Ok(Redirect::to(&format!("/?error={err}"))),
        }
    }

    // Validate CSRF state
    let saved_state: Option<String> = session.get("discord_oauth_state").await
        .unwrap_or(None);
    let param_state = params.state.as_deref().unwrap_or("");
    if saved_state.as_deref() != Some(param_state) || param_state.is_empty() {
        return Ok(Redirect::to("/?error=invalid_state"));
    }
    let _ = session.remove::<String>("discord_oauth_state").await;

    let code = params.code.as_deref()
        .ok_or_else(|| AppError::BadRequest("Missing authorization code".into()))?;

    // Exchange code for access token
    let token = exchange_code(&settings, code).await
        .map_err(|e| {
            tracing::error!("Discord token exchange failed: {e}");
            AppError::Internal(anyhow::anyhow!("Discord auth failed"))
        })?;

    // Get Discord user info
    let discord_user = get_discord_user(&token).await
        .map_err(|e| {
            tracing::error!("Discord user fetch failed: {e}");
            AppError::Internal(anyhow::anyhow!("Discord auth failed"))
        })?;

    // Check guild membership if required
    if !settings.guild_ids.is_empty() {
        let user_guilds = get_user_guilds(&token).await
            .map_err(|e| {
                tracing::error!("Discord guilds fetch failed: {e}");
                AppError::Internal(anyhow::anyhow!("Discord auth failed"))
            })?;

        let in_required_guild = user_guilds.iter()
            .any(|g| settings.guild_ids.contains(g));

        if !in_required_guild {
            return Ok(Redirect::to("/?error=guild_requirement_not_met"));
        }
    }

    // Check registration policy
    let existing = queries::get_user_by_discord_id(&state.db, &discord_user.id).await?;
    if existing.is_none() && !settings.allow_registration {
        return Ok(Redirect::to("/?error=registration_disabled"));
    }

    // Check user limit for new users
    if existing.is_none() {
        let app_settings = queries::get_app_settings(&state.db).await?;
        if let Some(limit) = app_settings.user_limit_total {
            let count = queries::count_users(&state.db).await?;
            if count >= limit {
                return Ok(Redirect::to("/?error=user_limit_reached"));
            }
        }
    }

    // Upsert user
    let (user_id, _created) = queries::upsert_discord_user(
        &state.db,
        &discord_user.id,
        &discord_user.username,
    ).await?;

    // Create session
    session.clear().await;
    session.insert(SESSION_USER_KEY, user_id).await
        .map_err(|e| AppError::Internal(anyhow::anyhow!("Session error: {e}")))?;

    let _ = queries::update_user_last_active(&state.db, user_id).await;

    tracing::info!("Discord login successful for user {user_id} (discord: {})", discord_user.username);
    Ok(Redirect::to("/"))
}

/// GET /api/auth/discord/config — admin-only.
async fn discord_config(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Json<Value>, AppError> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let db_settings = queries::get_discord_settings(&state.db).await?;
    let settings = load_discord_settings(&state).await?;

    // Build env_overrides (matches Python discord_routes.py)
    let mut env_overrides = serde_json::Map::new();
    if state.config.discord_client_id.is_some() {
        let mut e = serde_json::Map::new();
        e.insert("env_var".into(), json!("DISCORD_CLIENT_ID"));
        env_overrides.insert("client_id".into(), Value::Object(e));
    }
    if state.config.discord_client_secret.is_some() {
        let mut e = serde_json::Map::new();
        e.insert("env_var".into(), json!("DISCORD_CLIENT_SECRET"));
        e.insert("is_secret".into(), json!("true"));
        env_overrides.insert("client_secret".into(), Value::Object(e));
    }
    if let Some(ref uri) = state.config.discord_redirect_uri {
        let mut e = serde_json::Map::new();
        e.insert("env_var".into(), json!("DISCORD_REDIRECT_URI"));
        e.insert("value".into(), json!(uri));
        env_overrides.insert("redirect_uri".into(), Value::Object(e));
    }
    if let Some(ref gids) = state.config.discord_guild_ids {
        let mut e = serde_json::Map::new();
        e.insert("env_var".into(), json!("DISCORD_GUILD_IDS"));
        e.insert("value".into(), json!(gids));
        env_overrides.insert("guild_ids".into(), Value::Object(e));
    }
    if state.config.discord_allow_registration.is_some() {
        let mut e = serde_json::Map::new();
        e.insert("env_var".into(), json!("DISCORD_ALLOW_REGISTRATION"));
        e.insert("value".into(), json!(std::env::var("DISCORD_ALLOW_REGISTRATION").unwrap_or_default()));
        env_overrides.insert("allow_registration".into(), Value::Object(e));
    }

    // Mask client_secret like Python (client_secret_preview instead of client_secret_set)
    let secret_preview: Option<String> = db_settings
        .client_secret
        .as_ref()
        .and_then(|s| {
            let s = s.trim();
            if s.is_empty() { return None; }
            if s.len() <= 8 { return Some("****".into()); }
            Some(format!("{}...{}", &s[..4], &s[s.len() - 4..]))
        });

    // guild_ids: Python returns comma-joined string or ""
    let guild_ids_str = db_settings.guild_ids.as_deref().unwrap_or("");

    Ok(Json(json!({
        "config": {
            "enabled": settings.enabled,
            "client_id": db_settings.client_id,
            "client_secret_preview": secret_preview,
            "redirect_uri": db_settings.redirect_uri,
            "guild_ids": guild_ids_str,
            "allow_registration": db_settings.allow_registration,
        },
        "env_overrides": Value::Object(env_overrides),
    })))
}

#[derive(Deserialize)]
struct DiscordConfigUpdate {
    client_id: Option<String>,
    client_secret: Option<String>,
    redirect_uri: Option<String>,
    guild_ids: Option<String>,
    allow_registration: Option<bool>,
}

/// PUT /api/auth/discord/config — admin-only.
async fn discord_config_update(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<DiscordConfigUpdate>,
) -> Result<Json<Value>, AppError> {
    require_admin_user(&auth_user, state.config.require_auth)?;

    let current = queries::get_discord_settings(&state.db).await?;

    queries::update_discord_settings(
        &state.db,
        body.client_id.as_deref().or(current.client_id.as_deref()),
        body.client_secret.as_deref().or(current.client_secret.as_deref()),
        body.redirect_uri.as_deref().or(current.redirect_uri.as_deref()),
        body.guild_ids.as_deref().or(current.guild_ids.as_deref()),
        body.allow_registration.unwrap_or(current.allow_registration),
    ).await?;

    Ok(Json(json!({"status": "ok"})))
}

// ── Discord API helpers ──

struct DiscordToken {
    access_token: String,
}

#[derive(Debug)]
struct DiscordUser {
    id: String,
    username: String,
}

async fn exchange_code(settings: &ResolvedDiscordSettings, code: &str) -> anyhow::Result<DiscordToken> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://discord.com/api/oauth2/token")
        .form(&[
            ("client_id", settings.client_id.as_str()),
            ("client_secret", settings.client_secret.as_str()),
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", settings.redirect_uri.as_str()),
        ])
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Discord token exchange failed (HTTP {status}): {body}");
    }

    let data: Value = resp.json().await?;
    let access_token = data["access_token"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No access_token in Discord response"))?
        .to_string();

    Ok(DiscordToken { access_token })
}

async fn get_discord_user(token: &DiscordToken) -> anyhow::Result<DiscordUser> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://discord.com/api/v10/users/@me")
        .header("Authorization", format!("Bearer {}", token.access_token))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Discord user fetch failed (HTTP {status}): {body}");
    }

    let data: Value = resp.json().await?;
    let id = data["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No id in Discord user response"))?
        .to_string();
    let username = data["username"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No username in Discord user response"))?
        .to_string();

    Ok(DiscordUser { id, username })
}

async fn get_user_guilds(token: &DiscordToken) -> anyhow::Result<Vec<String>> {
    let client = reqwest::Client::new();
    let resp = client
        .get("https://discord.com/api/v10/users/@me/guilds")
        .header("Authorization", format!("Bearer {}", token.access_token))
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Discord guilds fetch failed (HTTP {status}): {body}");
    }

    let data: Vec<Value> = resp.json().await?;
    let guild_ids: Vec<String> = data
        .iter()
        .filter_map(|g| g["id"].as_str().map(|s| s.to_string()))
        .collect();

    Ok(guild_ids)
}

fn urlencoded(s: &str) -> String {
    form_urlencoded::byte_serialize(s.as_bytes()).collect()
}
