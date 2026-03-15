mod api;
mod audio;
mod auth;
mod classification;
mod config;
mod db;
mod error;
mod feeds;
mod jobs;
mod llm;
mod refinement;
mod transcription;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::http::HeaderValue;
use axum::middleware as axum_mw;
use axum::Router;
use sqlx::SqlitePool;
use tokio::sync::Mutex;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tower_sessions::cookie::time::Duration;
use tower_sessions::{Expiry, SessionManagerLayer};
use tracing_subscriber::EnvFilter;

use crate::auth::middleware::SharedRateLimiter;
use crate::auth::rate_limiter::FailureRateLimiter;
use crate::config::AppConfig;
use crate::jobs::manager::JobsManager;

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
    pub config: Arc<AppConfig>,
    pub rate_limiter: SharedRateLimiter,
    pub jobs_manager: Arc<JobsManager>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = AppConfig::from_env();
    tracing::info!("Starting Podly on {}:{}", config.host, config.port);

    // Ensure data directory exists
    tokio::fs::create_dir_all(&config.data_dir).await?;

    // Set up database
    let pool = db::create_pool(&config.database_url).await?;
    db::run_migrations(&pool).await?;
    db::seed_settings(&pool).await?;

    // Create default admin user if configured and no users exist
    if let (Some(username), Some(password)) = (
        &config.default_admin_username,
        &config.default_admin_password,
    ) {
        let user_count = db::queries::count_users(&pool).await?;
        if user_count == 0 {
            let hash = auth::hash_password(password)
                .map_err(|e| anyhow::anyhow!("Failed to hash default admin password: {e}"))?;
            db::queries::insert_user(&pool, username, &hash, "admin").await?;
            tracing::info!("Created default admin user: {username}");
        }
    }

    // Set up SQLite-backed session store (persists across restarts)
    let session_store = db::session_store::SqliteSessionStore::new(pool.clone())
        .await
        .expect("Failed to initialize session store");

    let session_layer = SessionManagerLayer::new(session_store)
        .with_expiry(Expiry::OnInactivity(Duration::days(30)))
        .with_secure(false); // Allow HTTP for local dev

    let rate_limiter: SharedRateLimiter = Arc::new(Mutex::new(FailureRateLimiter::new()));

    // Build CORS layer from CORS_ORIGINS env var (comma-separated) or default to permissive
    let cors_layer = if let Some(origins) = &config.cors_origins {
        let origin_list: Vec<HeaderValue> = origins
            .split(',')
            .filter_map(|o| o.trim().parse().ok())
            .collect();
        if origin_list.is_empty() {
            CorsLayer::permissive()
        } else {
            CorsLayer::new()
                .allow_origin(AllowOrigin::list(origin_list))
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any)
                .allow_credentials(true)
        }
    } else {
        CorsLayer::permissive()
    };

    let config = Arc::new(config);
    let jobs_manager = Arc::new(JobsManager::new(pool.clone(), Arc::clone(&config)));

    let state = AppState {
        db: pool,
        config: Arc::clone(&config),
        rate_limiter,
        jobs_manager: jobs_manager.clone(),
    };

    // Clean up stale/zombie jobs from previous run (matches Python's clear_all_jobs on startup)
    jobs_manager.cleanup_on_startup().await;

    // Start the jobs manager background worker
    jobs_manager.start();

    // Start the periodic scheduler (unless disabled via PODLY_DISABLE_SCHEDULER)
    if !config.disable_scheduler {
        jobs::scheduler::start_scheduler(jobs_manager.clone(), state.db.clone()).await;
    } else {
        tracing::info!("Scheduler disabled via PODLY_DISABLE_SCHEDULER");
    }

    // Build router
    let static_dir = config.static_dir.clone();
    let app = Router::new()
        .merge(api::router())
        .layer(axum_mw::from_fn_with_state(
            state.clone(),
            auth::middleware::auth_middleware,
        ))
        .layer(session_layer)
        .fallback_service(
            ServeDir::new(&static_dir).fallback(ServeFile::new(static_dir.join("index.html"))),
        )
        .layer(TraceLayer::new_for_http())
        .layer(cors_layer)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    tracing::info!("Listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
