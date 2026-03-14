/// Legacy data migration tool
///
/// Migrates data from the Python/Flask Podly SQLite database to the new Rust backend schema.
/// The schemas are nearly identical, so this is mostly a straight copy with minor adjustments.
///
/// Usage:
///   migrate_legacy --old /path/to/old/sqlite3.db --new /path/to/new/podly.db
///
/// The new database must already have the schema created (run the main binary first to apply migrations).

use std::path::PathBuf;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};

#[derive(Debug)]
struct Args {
    old_db: PathBuf,
    new_db: PathBuf,
    skip_jobs: bool,
    dry_run: bool,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut old_db = None;
    let mut new_db = None;
    let mut skip_jobs = false;
    let mut dry_run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--old" | "-o" => {
                i += 1;
                old_db = Some(PathBuf::from(&args[i]));
            }
            "--new" | "-n" => {
                i += 1;
                new_db = Some(PathBuf::from(&args[i]));
            }
            "--skip-jobs" => skip_jobs = true,
            "--dry-run" => dry_run = true,
            "--help" | "-h" => {
                eprintln!("Usage: migrate_legacy --old <old.db> --new <new.db> [--skip-jobs] [--dry-run]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --old, -o     Path to old Python SQLite database");
                eprintln!("  --new, -n     Path to new Rust SQLite database (must have schema applied)");
                eprintln!("  --skip-jobs   Skip migrating processing_job and jobs_manager_run tables");
                eprintln!("  --dry-run     Report counts but don't actually migrate");
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Args {
        old_db: old_db.unwrap_or_else(|| {
            eprintln!("Missing --old argument. Use --help for usage.");
            std::process::exit(1);
        }),
        new_db: new_db.unwrap_or_else(|| {
            eprintln!("Missing --new argument. Use --help for usage.");
            std::process::exit(1);
        }),
        skip_jobs,
        dry_run,
    }
}

async fn open_pool(path: &PathBuf, read_only: bool) -> Result<SqlitePool, sqlx::Error> {
    let opts = SqliteConnectOptions::new()
        .filename(path)
        .read_only(read_only)
        .create_if_missing(false);

    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(opts)
        .await
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = parse_args();

    println!("=== Podly Legacy Migration Tool ===");
    println!("Old DB: {}", args.old_db.display());
    println!("New DB: {}", args.new_db.display());
    if args.dry_run {
        println!("Mode: DRY RUN (no data will be written)");
    }
    println!();

    let old = open_pool(&args.old_db, true).await?;
    let new = open_pool(&args.new_db, false).await?;

    // Migrate in dependency order
    migrate_feeds(&old, &new, args.dry_run).await?;
    migrate_posts(&old, &new, args.dry_run).await?;
    migrate_users(&old, &new, args.dry_run).await?;
    migrate_transcript_segments(&old, &new, args.dry_run).await?;
    migrate_model_calls(&old, &new, args.dry_run).await?;
    migrate_identifications(&old, &new, args.dry_run).await?;
    migrate_feed_supporters(&old, &new, args.dry_run).await?;
    migrate_feed_access_tokens(&old, &new, args.dry_run).await?;

    if !args.skip_jobs {
        migrate_jobs_manager_runs(&old, &new, args.dry_run).await?;
        migrate_processing_jobs(&old, &new, args.dry_run).await?;
    } else {
        println!("[SKIP] jobs_manager_run (--skip-jobs)");
        println!("[SKIP] processing_job (--skip-jobs)");
    }

    // Settings tables
    migrate_settings_table(&old, &new, "llm_settings", args.dry_run).await?;
    migrate_settings_table(&old, &new, "whisper_settings", args.dry_run).await?;
    migrate_settings_table(&old, &new, "processing_settings", args.dry_run).await?;
    migrate_settings_table(&old, &new, "output_settings", args.dry_run).await?;
    migrate_settings_table(&old, &new, "app_settings", args.dry_run).await?;
    migrate_settings_table(&old, &new, "discord_settings", args.dry_run).await?;
    migrate_settings_table(&old, &new, "chapter_filter_settings", args.dry_run).await?;

    println!();
    println!("=== Migration complete ===");
    if args.dry_run {
        println!("(dry run — no data was written)");
    }

    Ok(())
}

async fn migrate_feeds(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM feed ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[feed] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO feed (id, alt_id, title, description, author, rss_url, image_url, ad_detection_strategy, chapter_filter_strings, auto_whitelist_new_episodes_override)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<Option<String>, _>("alt_id"))
        .bind(row.get::<String, _>("title"))
        .bind(row.get::<Option<String>, _>("description"))
        .bind(row.get::<Option<String>, _>("author"))
        .bind(row.get::<String, _>("rss_url"))
        .bind(row.get::<Option<String>, _>("image_url"))
        .bind(row.get::<String, _>("ad_detection_strategy"))
        .bind(row.get::<Option<String>, _>("chapter_filter_strings"))
        .bind(row.get::<Option<bool>, _>("auto_whitelist_new_episodes_override"))
        .execute(new)
        .await?;
    }
    println!("[feed] Migrated {count} rows");
    Ok(())
}

async fn migrate_posts(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM post ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[post] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO post (id, feed_id, guid, download_url, title, unprocessed_audio_path, processed_audio_path, description, release_date, duration, whitelisted, image_url, download_count, chapter_data, refined_ad_boundaries, refined_ad_boundaries_updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<i64, _>("feed_id"))
        .bind(row.get::<String, _>("guid"))
        .bind(row.get::<String, _>("download_url"))
        .bind(row.get::<String, _>("title"))
        .bind(row.get::<Option<String>, _>("unprocessed_audio_path"))
        .bind(row.get::<Option<String>, _>("processed_audio_path"))
        .bind(row.get::<Option<String>, _>("description"))
        .bind(row.get::<Option<String>, _>("release_date"))
        .bind(row.get::<Option<i64>, _>("duration"))
        .bind(row.get::<bool, _>("whitelisted"))
        .bind(row.get::<Option<String>, _>("image_url"))
        .bind(row.get::<Option<i64>, _>("download_count"))
        .bind(row.get::<Option<String>, _>("chapter_data"))
        .bind(row.get::<Option<String>, _>("refined_ad_boundaries"))
        .bind(row.get::<Option<String>, _>("refined_ad_boundaries_updated_at"))
        .execute(new)
        .await?;
    }
    println!("[post] Migrated {count} rows");
    Ok(())
}

async fn migrate_users(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM users ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[users] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    let mut bcrypt_count = 0;
    for row in &rows {
        let password_hash: String = row.get("password_hash");
        if password_hash.starts_with("$2") {
            bcrypt_count += 1;
        }
        // Preserve original hashes — users with bcrypt hashes will need to reset passwords
        // since the Rust backend uses argon2
        sqlx::query(
            "INSERT OR IGNORE INTO users (id, username, password_hash, role, feed_allowance, feed_subscription_status, stripe_customer_id, stripe_subscription_id, created_at, updated_at, discord_id, discord_username, last_active, manual_feed_allowance)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<String, _>("username"))
        .bind(&password_hash)
        .bind(row.get::<String, _>("role"))
        .bind(row.get::<i64, _>("feed_allowance"))
        .bind(row.get::<String, _>("feed_subscription_status"))
        .bind(row.get::<Option<String>, _>("stripe_customer_id"))
        .bind(row.get::<Option<String>, _>("stripe_subscription_id"))
        .bind(row.get::<String, _>("created_at"))
        .bind(row.get::<String, _>("updated_at"))
        .bind(row.get::<Option<String>, _>("discord_id"))
        .bind(row.get::<Option<String>, _>("discord_username"))
        .bind(row.get::<Option<String>, _>("last_active"))
        .bind(row.get::<Option<i64>, _>("manual_feed_allowance"))
        .execute(new)
        .await?;
    }
    println!("[users] Migrated {count} rows");
    if bcrypt_count > 0 {
        println!("[users] WARNING: {bcrypt_count} users have bcrypt password hashes.");
        println!("         These users will need to reset their passwords (Rust backend uses argon2).");
    }
    Ok(())
}

async fn migrate_transcript_segments(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM transcript_segment ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[transcript_segment] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO transcript_segment (id, post_id, sequence_num, start_time, end_time, text)
             VALUES (?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<i64, _>("post_id"))
        .bind(row.get::<i64, _>("sequence_num"))
        .bind(row.get::<f64, _>("start_time"))
        .bind(row.get::<f64, _>("end_time"))
        .bind(row.get::<String, _>("text"))
        .execute(new)
        .await?;
    }
    println!("[transcript_segment] Migrated {count} rows");
    Ok(())
}

async fn migrate_model_calls(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM model_call ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[model_call] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO model_call (id, post_id, first_segment_sequence_num, last_segment_sequence_num, model_name, prompt, response, timestamp, status, error_message, retry_attempts)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<i64, _>("post_id"))
        .bind(row.get::<i64, _>("first_segment_sequence_num"))
        .bind(row.get::<i64, _>("last_segment_sequence_num"))
        .bind(row.get::<String, _>("model_name"))
        .bind(row.get::<String, _>("prompt"))
        .bind(row.get::<Option<String>, _>("response"))
        .bind(row.get::<String, _>("timestamp"))
        .bind(row.get::<String, _>("status"))
        .bind(row.get::<Option<String>, _>("error_message"))
        .bind(row.get::<i64, _>("retry_attempts"))
        .execute(new)
        .await?;
    }
    println!("[model_call] Migrated {count} rows");
    Ok(())
}

async fn migrate_identifications(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM identification ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[identification] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO identification (id, transcript_segment_id, model_call_id, confidence, label)
             VALUES (?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<i64, _>("transcript_segment_id"))
        .bind(row.get::<i64, _>("model_call_id"))
        .bind(row.get::<Option<f64>, _>("confidence"))
        .bind(row.get::<String, _>("label"))
        .execute(new)
        .await?;
    }
    println!("[identification] Migrated {count} rows");
    Ok(())
}

async fn migrate_feed_supporters(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM feed_supporter ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[feed_supporter] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO feed_supporter (id, feed_id, user_id, created_at)
             VALUES (?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<i64, _>("feed_id"))
        .bind(row.get::<i64, _>("user_id"))
        .bind(row.get::<String, _>("created_at"))
        .execute(new)
        .await?;
    }
    println!("[feed_supporter] Migrated {count} rows");
    Ok(())
}

async fn migrate_feed_access_tokens(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM feed_access_token ORDER BY id").fetch_all(old).await?;
    let count = rows.len();
    println!("[feed_access_token] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO feed_access_token (id, token_id, token_hash, token_secret, feed_id, user_id, created_at, last_used_at, revoked)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<i64, _>("id"))
        .bind(row.get::<String, _>("token_id"))
        .bind(row.get::<String, _>("token_hash"))
        .bind(row.get::<Option<String>, _>("token_secret"))
        .bind(row.get::<Option<i64>, _>("feed_id"))
        .bind(row.get::<i64, _>("user_id"))
        .bind(row.get::<String, _>("created_at"))
        .bind(row.get::<Option<String>, _>("last_used_at"))
        .bind(row.get::<bool, _>("revoked"))
        .execute(new)
        .await?;
    }
    println!("[feed_access_token] Migrated {count} rows");
    Ok(())
}

async fn migrate_jobs_manager_runs(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM jobs_manager_run ORDER BY created_at").fetch_all(old).await?;
    let count = rows.len();
    println!("[jobs_manager_run] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO jobs_manager_run (id, status, trigger, started_at, completed_at, total_jobs, queued_jobs, running_jobs, completed_jobs, failed_jobs, skipped_jobs, context_json, counters_reset_at, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<String, _>("id"))
        .bind(row.get::<String, _>("status"))
        .bind(row.get::<String, _>("trigger"))
        .bind(row.get::<Option<String>, _>("started_at"))
        .bind(row.get::<Option<String>, _>("completed_at"))
        .bind(row.get::<i64, _>("total_jobs"))
        .bind(row.get::<i64, _>("queued_jobs"))
        .bind(row.get::<i64, _>("running_jobs"))
        .bind(row.get::<i64, _>("completed_jobs"))
        .bind(row.get::<i64, _>("failed_jobs"))
        .bind(row.get::<i64, _>("skipped_jobs"))
        .bind(row.get::<Option<String>, _>("context_json"))
        .bind(row.get::<Option<String>, _>("counters_reset_at"))
        .bind(row.get::<Option<String>, _>("created_at"))
        .bind(row.get::<Option<String>, _>("updated_at"))
        .execute(new)
        .await?;
    }
    println!("[jobs_manager_run] Migrated {count} rows");
    Ok(())
}

async fn migrate_processing_jobs(old: &SqlitePool, new: &SqlitePool, dry_run: bool) -> anyhow::Result<()> {
    let rows: Vec<sqlx::sqlite::SqliteRow> =
        sqlx::query("SELECT * FROM processing_job ORDER BY created_at").fetch_all(old).await?;
    let count = rows.len();
    println!("[processing_job] Found {count} rows");
    if dry_run || count == 0 {
        return Ok(());
    }

    for row in &rows {
        sqlx::query(
            "INSERT OR IGNORE INTO processing_job (id, jobs_manager_run_id, post_guid, status, current_step, step_name, total_steps, progress_percentage, started_at, completed_at, error_message, scheduler_job_id, created_at, requested_by_user_id, billing_user_id)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(row.get::<String, _>("id"))
        .bind(row.get::<Option<String>, _>("jobs_manager_run_id"))
        .bind(row.get::<String, _>("post_guid"))
        .bind(row.get::<String, _>("status"))
        .bind(row.get::<Option<i64>, _>("current_step"))
        .bind(row.get::<Option<String>, _>("step_name"))
        .bind(row.get::<Option<i64>, _>("total_steps"))
        .bind(row.get::<Option<f64>, _>("progress_percentage"))
        .bind(row.get::<Option<String>, _>("started_at"))
        .bind(row.get::<Option<String>, _>("completed_at"))
        .bind(row.get::<Option<String>, _>("error_message"))
        .bind(row.get::<Option<String>, _>("scheduler_job_id"))
        .bind(row.get::<Option<String>, _>("created_at"))
        .bind(row.get::<Option<i64>, _>("requested_by_user_id"))
        .bind(row.get::<Option<i64>, _>("billing_user_id"))
        .execute(new)
        .await?;
    }
    println!("[processing_job] Migrated {count} rows");
    Ok(())
}

/// Migrate a settings singleton table.
/// Since these are single-row tables and columns may differ slightly between old/new,
/// we use a generic approach: read column names from old, match to new, copy what overlaps.
async fn migrate_settings_table(old: &SqlitePool, new: &SqlitePool, table: &str, dry_run: bool) -> anyhow::Result<()> {
    // Get column info from both databases
    let old_cols: Vec<(String,)> = sqlx::query_as(&format!("SELECT name FROM pragma_table_info('{table}')"))
        .fetch_all(old)
        .await?;
    let new_cols: Vec<(String,)> = sqlx::query_as(&format!("SELECT name FROM pragma_table_info('{table}')"))
        .fetch_all(new)
        .await?;

    let old_col_names: Vec<&str> = old_cols.iter().map(|c| c.0.as_str()).collect();
    let new_col_names: Vec<&str> = new_cols.iter().map(|c| c.0.as_str()).collect();

    // Find common columns
    let common: Vec<&str> = old_col_names
        .iter()
        .filter(|c| new_col_names.contains(c))
        .copied()
        .collect();

    let only_old: Vec<&str> = old_col_names.iter().filter(|c| !new_col_names.contains(c)).copied().collect();
    let only_new: Vec<&str> = new_col_names.iter().filter(|c| !old_col_names.contains(c)).copied().collect();

    println!("[{table}] Common columns: {}, old-only: {:?}, new-only: {:?}", common.len(), only_old, only_new);

    if dry_run || common.is_empty() {
        return Ok(());
    }

    // Read the single row from old
    let col_list = common.join(", ");
    let row: Option<sqlx::sqlite::SqliteRow> =
        sqlx::query(&format!("SELECT {col_list} FROM {table} WHERE id = 1"))
            .fetch_optional(old)
            .await?;

    let Some(row) = row else {
        println!("[{table}] No settings row found in old database");
        return Ok(());
    };

    // Delete existing and insert
    sqlx::query(&format!("DELETE FROM {table} WHERE id = 1"))
        .execute(new)
        .await?;

    let placeholders: Vec<&str> = common.iter().map(|_| "?").collect();
    let insert_sql = format!(
        "INSERT INTO {table} ({col_list}) VALUES ({})",
        placeholders.join(", ")
    );

    let mut query = sqlx::query(&insert_sql);
    for col in &common {
        // Bind all as text (SQLite is flexible with types)
        let val: Option<String> = row.try_get::<Option<String>, _>(*col).unwrap_or(None);
        query = query.bind(val);
    }
    query.execute(new).await?;

    println!("[{table}] Migrated settings");
    Ok(())
}
