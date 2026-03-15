pub mod models;
#[allow(dead_code)]
pub mod queries;
pub mod session_store;

use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use std::str::FromStr;

pub async fn create_pool(database_url: &str) -> Result<SqlitePool, sqlx::Error> {
    let options = SqliteConnectOptions::from_str(database_url)?
        .journal_mode(SqliteJournalMode::Wal)
        .create_if_missing(true)
        .foreign_keys(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await?;

    Ok(pool)
}

pub async fn run_migrations(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let migration_sql = include_str!("../../migrations/001_initial.sql");

    for statement in migration_sql.split(';') {
        // Strip comment lines, then check if anything remains
        let cleaned: String = statement
            .lines()
            .filter(|line| {
                let t = line.trim();
                !t.is_empty() && !t.starts_with("--")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let trimmed = cleaned.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Skip PRAGMAs — already set via connection options
        if trimmed.to_uppercase().starts_with("PRAGMA") {
            continue;
        }

        sqlx::query(trimmed).execute(pool).await?;
    }

    // Schema evolution: add columns that may not exist in older databases
    let alter_statements = [
        "ALTER TABLE processing_settings ADD COLUMN max_overlap_segments INTEGER NOT NULL DEFAULT 30",
    ];
    for stmt in alter_statements {
        // SQLite returns an error if column already exists — ignore it
        let _ = sqlx::query(stmt).execute(pool).await;
    }

    tracing::info!("Database migrations complete");
    Ok(())
}

pub async fn seed_settings(pool: &SqlitePool) -> Result<(), sqlx::Error> {
    let now = chrono::Utc::now().to_rfc3339();

    // Seed singleton settings rows if they don't exist
    let settings_tables = [
        "llm_settings",
        "whisper_settings",
        "processing_settings",
        "output_settings",
        "app_settings",
        "discord_settings",
        "chapter_filter_settings",
    ];

    for table in settings_tables {
        let exists: Option<(i64,)> =
            sqlx::query_as(&format!("SELECT id FROM {table} WHERE id = 1"))
                .fetch_optional(pool)
                .await?;

        if exists.is_none() {
            sqlx::query(&format!(
                "INSERT INTO {table} (id, created_at, updated_at) VALUES (1, ?, ?)"
            ))
            .bind(&now)
            .bind(&now)
            .execute(pool)
            .await?;
            tracing::info!("Seeded {table} with defaults");
        }
    }

    Ok(())
}
