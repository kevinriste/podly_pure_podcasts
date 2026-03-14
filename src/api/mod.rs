pub mod auth;
pub mod billing;
pub mod config;
pub mod discord;
pub mod feeds;
pub mod jobs;
pub mod posts;

use axum::Router;

use crate::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .merge(auth::router())
        .merge(feeds::router())
        .merge(posts::router())
        .merge(jobs::router())
        .merge(config::router())
        .merge(discord::router())
        .merge(billing::router())
}
