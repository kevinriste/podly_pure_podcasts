use std::path::{Path, PathBuf};

use sqlx::SqlitePool;

use crate::classification::classifier::{self, ClassifierConfig, IdentifiedAd, Segment};
use crate::config::AppConfig;

/// Run the full processing pipeline for a post.
///
/// Steps:
/// 1. Download - fetch episode audio from URL
/// 2. Transcribe - convert audio to text segments
/// 3. Classify - identify ad segments via LLM
/// 4. Refine - adjust ad boundaries
/// 5. Cut - remove ad segments from audio via ffmpeg
/// 6. Finalize - update DB with results
pub async fn run_pipeline(
    pool: &SqlitePool,
    config: &AppConfig,
    job_id: &str,
    post_guid: &str,
) -> Result<(), PipelineError> {
    let post: Option<(i64, i64, String, Option<String>, Option<String>, Option<String>)> =
        sqlx::query_as(
            "SELECT id, feed_id, title, download_url, unprocessed_audio_path, processed_audio_path FROM post WHERE guid = ?",
        )
        .bind(post_guid)
        .fetch_optional(pool)
        .await
        .map_err(|e| PipelineError::Db(e.to_string()))?;

    let Some((post_id, feed_id, post_title, download_url, existing_audio, processed_audio)) = post
    else {
        return Err(PipelineError::NotFound("Post not found".into()));
    };

    if let Some(processed) = &processed_audio {
        if Path::new(processed).exists() {
            update_step(pool, job_id, 6, "already_processed", 100.0).await;
            return Ok(());
        }
    }

    let download_url = download_url
        .ok_or_else(|| PipelineError::Validation("No download URL".into()))?;

    let feed_info: Option<(String, Option<String>, String)> =
        sqlx::query_as("SELECT title, description, ad_detection_strategy FROM feed WHERE id = ?")
            .bind(feed_id)
            .fetch_optional(pool)
            .await
            .unwrap_or(None);
    let (feed_title, feed_description, feed_strategy) = feed_info
        .unwrap_or_else(|| ("Unknown Podcast".into(), None, "inherit".into()));
    let feed_description = feed_description.unwrap_or_else(|| feed_title.clone());

    // Determine ad detection strategy: feed-level overrides app-level
    let app_strategy: String = sqlx::query_as::<_, (String,)>(
        "SELECT ad_detection_strategy FROM app_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None)
    .map(|(s,)| s)
    .unwrap_or_else(|| "llm".into());
    let strategy = if feed_strategy == "inherit" { &app_strategy } else { &feed_strategy };

    // Load chapter_filter_strings: per-feed override, or global default
    let feed_filter: Option<String> = sqlx::query_as::<_, (Option<String>,)>(
        "SELECT chapter_filter_strings FROM feed WHERE id = ?",
    )
    .bind(feed_id)
    .fetch_optional(pool)
    .await
    .unwrap_or(None)
    .and_then(|(s,)| s);
    let chapter_filter_strings = if feed_filter.is_some() {
        feed_filter
    } else {
        sqlx::query_as::<_, (Option<String>,)>(
            "SELECT default_filter_strings FROM chapter_filter_settings WHERE id = 1",
        )
        .fetch_optional(pool)
        .await
        .unwrap_or(None)
        .and_then(|(s,)| s)
    };

    // Step 1: Download
    update_step(pool, job_id, 1, "downloading", 0.0).await;
    let audio_path = if let Some(existing) = &existing_audio {
        if Path::new(existing).exists() {
            PathBuf::from(existing)
        } else {
            download_audio(pool, post_id, &download_url, &post_title, job_id).await?
        }
    } else {
        download_audio(pool, post_id, &download_url, &post_title, job_id).await?
    };
    update_step(pool, job_id, 1, "downloaded", 16.0).await;

    // Strategy-dependent steps
    let refined = match strategy.as_str() {
        "chapter" => {
            // Chapter-based: skip transcription + LLM, read chapters from audio
            update_step(pool, job_id, 2, "reading_chapters", 20.0).await;
            let chapter_ads = classify_chapters(&audio_path, chapter_filter_strings.as_deref()).await?;
            update_step(pool, job_id, 2, "chapters_read", 40.0).await;
            update_step(pool, job_id, 3, "chapters_filtered", 60.0).await;
            update_step(pool, job_id, 4, "skipped", 70.0).await;
            chapter_ads
        }
        _ => {
            // LLM-based or oneshot: transcribe, classify, refine
            update_step(pool, job_id, 2, "transcribing", 20.0).await;
            let segments = transcribe(pool, config, post_id, &audio_path).await?;
            update_step(pool, job_id, 2, "transcribed", 40.0).await;

            update_step(pool, job_id, 3, "classifying", 45.0).await;
            let ad_segments = match strategy.as_str() {
                "oneshot" => classify_oneshot(pool, config, post_id, &segments, &feed_title, &feed_description).await?,
                _ => classify(pool, config, post_id, &segments, &feed_title, &feed_description).await?,
            };
            update_step(pool, job_id, 3, "classified", 60.0).await;

            update_step(pool, job_id, 4, "refining", 65.0).await;
            let refined = if strategy.as_str() == "oneshot" {
                ad_segments.iter().map(|a| (a.start_time, a.end_time)).collect()
            } else {
                refine(pool, config, post_id, &ad_segments, &segments).await
            };
            update_step(pool, job_id, 4, "refined", 70.0).await;
            refined
        }
    };

    // Step 5: Cut
    update_step(pool, job_id, 5, "cutting", 75.0).await;
    let output_path = cut_audio(&audio_path, &refined, &feed_title, &post_title).await?;
    update_step(pool, job_id, 5, "cut", 90.0).await;

    // Step 6: Finalize
    update_step(pool, job_id, 6, "finalizing", 95.0).await;

    let _ = sqlx::query("UPDATE post SET processed_audio_path = ? WHERE id = ?")
        .bind(output_path.to_str().unwrap_or(""))
        .bind(post_id)
        .execute(pool)
        .await;

    if let Some(existing) = &existing_audio {
        let _ = tokio::fs::remove_file(existing).await;
        let _ = sqlx::query("UPDATE post SET unprocessed_audio_path = NULL WHERE id = ?")
            .bind(post_id)
            .execute(pool)
            .await;
    }

    if !refined.is_empty() {
        let boundaries_json = serde_json::to_string(&refined).unwrap_or_default();
        let now = chrono::Utc::now().to_rfc3339();
        let _ = sqlx::query(
            "UPDATE post SET refined_ad_boundaries = ?, refined_ad_boundaries_updated_at = ? WHERE id = ?",
        )
        .bind(&boundaries_json)
        .bind(&now)
        .bind(post_id)
        .execute(pool)
        .await;
    }

    update_step(pool, job_id, 6, "complete", 100.0).await;
    Ok(())
}

async fn update_step(pool: &SqlitePool, job_id: &str, step: i64, name: &str, progress: f64) {
    let _ = sqlx::query(
        "UPDATE processing_job SET current_step = ?, step_name = ?, progress_percentage = ? WHERE id = ?",
    )
    .bind(step)
    .bind(name)
    .bind(progress)
    .bind(job_id)
    .execute(pool)
    .await;
}

async fn download_audio(
    pool: &SqlitePool,
    post_id: i64,
    url: &str,
    post_title: &str,
    job_id: &str,
) -> Result<PathBuf, PipelineError> {
    let sanitized = sanitize_filename(post_title);
    let dir = PathBuf::from("data/in").join(&sanitized);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| PipelineError::Io(e.to_string()))?;

    let url_filename = url
        .rsplit('/')
        .next()
        .unwrap_or("episode.mp3")
        .split('?')
        .next()
        .unwrap_or("episode.mp3");
    let filename = format!("{job_id}_{url_filename}");
    let path = dir.join(&filename);

    tracing::info!("Downloading audio from {url} to {}", path.display());

    let response = reqwest::get(url)
        .await
        .map_err(|e| PipelineError::Download(e.to_string()))?;

    if !response.status().is_success() {
        return Err(PipelineError::Download(format!(
            "HTTP {} for {url}",
            response.status()
        )));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| PipelineError::Download(e.to_string()))?;

    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|e| PipelineError::Io(e.to_string()))?;

    let path_str = path.to_str().unwrap_or("");
    let _ = sqlx::query("UPDATE post SET unprocessed_audio_path = ? WHERE id = ?")
        .bind(path_str)
        .bind(post_id)
        .execute(pool)
        .await;

    Ok(path)
}

async fn transcribe(
    pool: &SqlitePool,
    config: &AppConfig,
    post_id: i64,
    audio_path: &Path,
) -> Result<Vec<Segment>, PipelineError> {
    // Check for existing segments
    let existing: Vec<(i64, i64, f64, f64, String)> = sqlx::query_as(
        "SELECT id, sequence_num, start_time, end_time, text FROM transcript_segment WHERE post_id = ? ORDER BY sequence_num",
    )
    .bind(post_id)
    .fetch_all(pool)
    .await
    .map_err(|e| PipelineError::Db(e.to_string()))?;

    if !existing.is_empty() {
        tracing::info!("Using {} existing transcript segments for post {post_id}", existing.len());
        return Ok(existing
            .into_iter()
            .map(|(id, seq, start, end, text)| Segment {
                id,
                sequence_num: seq,
                start_time: start,
                end_time: end,
                text,
            })
            .collect());
    }

    // Load whisper settings from DB
    let whisper: Option<(String, String, Option<String>, String, String, i64, i64, Option<String>, String, String, i64)> =
        sqlx::query_as(
            "SELECT whisper_type, local_model, remote_api_key, remote_base_url, remote_model, remote_timeout_sec, remote_chunksize_mb, groq_api_key, groq_model, groq_language, groq_max_retries FROM whisper_settings WHERE id = 1",
        )
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

    let Some((db_whisper_type, _local_model, db_remote_api_key, remote_base_url, remote_model, remote_timeout, remote_chunksize, db_groq_api_key, groq_model, groq_language, groq_max_retries)) = whisper else {
        return Err(PipelineError::Transcription("No whisper settings configured".into()));
    };

    // Apply env var overrides: env > DB
    let whisper_type = config.whisper_type.as_deref().unwrap_or(&db_whisper_type);
    let groq_api_key = config.groq_api_key.clone().or(db_groq_api_key);
    let remote_api_key = config.whisper_remote_api_key.clone().or(db_remote_api_key);

    let audio_path_str = audio_path.to_str().unwrap_or("");

    use crate::transcription::Transcriber;

    let result = match whisper_type {
        "groq" => {
            let api_key = groq_api_key
                .ok_or_else(|| PipelineError::Transcription("Groq API key not configured (set GROQ_API_KEY env or DB)".into()))?;
            let transcriber = crate::transcription::groq::GroqWhisperTranscriber::new(
                &groq_model, &api_key, &groq_language, groq_max_retries as u32,
            );
            transcriber.transcribe(audio_path_str).await
                .map_err(|e| PipelineError::Transcription(e.to_string()))?
        }
        "remote" => {
            let api_key = remote_api_key
                .ok_or_else(|| PipelineError::Transcription("Remote whisper API key not configured (set WHISPER_REMOTE_API_KEY env or DB)".into()))?;
            let transcriber = crate::transcription::remote::RemoteWhisperTranscriber::new(
                &remote_model, &remote_base_url, &api_key, &groq_language,
                remote_timeout as u64, remote_chunksize as usize,
            );
            transcriber.transcribe(audio_path_str).await
                .map_err(|e| PipelineError::Transcription(e.to_string()))?
        }
        #[cfg(feature = "local-whisper")]
        "local" => {
            let model_path = format!("models/ggml-{_local_model}.bin");
            let transcriber = crate::transcription::local::LocalWhisperTranscriber::new(
                &_local_model, &model_path,
            );
            transcriber.transcribe(audio_path_str).await
                .map_err(|e| PipelineError::Transcription(e.to_string()))?
        }
        other => {
            return Err(PipelineError::Transcription(format!(
                "Unknown whisper type: {other}"
            )));
        }
    };

    // Persist segments
    let mut db_segments = Vec::new();
    for (i, seg) in result.segments.iter().enumerate() {
        let seq = i as i64;
        let insert_result = sqlx::query(
            "INSERT INTO transcript_segment (post_id, sequence_num, start_time, end_time, text) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(post_id)
        .bind(seq)
        .bind(seg.start)
        .bind(seg.end)
        .bind(&seg.text)
        .execute(pool)
        .await
        .map_err(|e| PipelineError::Db(e.to_string()))?;

        db_segments.push(Segment {
            id: insert_result.last_insert_rowid(),
            sequence_num: seq,
            start_time: seg.start,
            end_time: seg.end,
            text: seg.text.clone(),
        });
    }

    tracing::info!("Transcribed {} segments for post {post_id}", db_segments.len());
    Ok(db_segments)
}

async fn classify(
    pool: &SqlitePool,
    config: &AppConfig,
    post_id: i64,
    segments: &[Segment],
    feed_title: &str,
    feed_description: &str,
) -> Result<Vec<IdentifiedAd>, PipelineError> {
    // Check for existing identifications
    let existing_count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM identification i
         JOIN transcript_segment ts ON ts.id = i.transcript_segment_id
         WHERE ts.post_id = ?",
    )
    .bind(post_id)
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    if existing_count.0 > 0 {
        let existing: Vec<(i64, f64, f64, f64, String)> = sqlx::query_as(
            "SELECT ts.id, ts.start_time, ts.end_time, i.confidence, i.label
             FROM identification i
             JOIN transcript_segment ts ON ts.id = i.transcript_segment_id
             WHERE ts.post_id = ? AND i.label = 'ad'
             ORDER BY ts.start_time",
        )
        .bind(post_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        return Ok(existing
            .into_iter()
            .map(|(id, start, end, conf, label)| IdentifiedAd {
                segment_id: id,
                start_time: start,
                end_time: end,
                confidence: conf,
                label,
            })
            .collect());
    }

    // Load LLM settings from DB
    let llm: Option<(Option<String>, String, Option<String>, i64, i64, i64, i64, f64)> =
        sqlx::query_as(
            "SELECT l.llm_api_key, l.llm_model, l.openai_base_url, l.openai_timeout, l.openai_max_tokens, l.llm_max_concurrent_calls, l.llm_max_retry_attempts, o.min_confidence
             FROM llm_settings l, output_settings o WHERE l.id = 1 AND o.id = 1",
        )
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

    let Some((db_api_key, db_model, db_base_url, timeout, max_tokens, max_concurrent, max_retries, min_confidence)) = llm else {
        tracing::warn!("LLM settings not configured — skipping classification");
        return Ok(vec![]);
    };

    // Apply env var overrides: env > DB
    let api_key = config.llm_api_key.clone()
        .or(db_api_key)
        .filter(|k| !k.is_empty());
    let api_key = match api_key {
        Some(k) => k,
        None => {
            tracing::warn!("No LLM API key configured (set LLM_API_KEY env or DB) — skipping classification");
            return Ok(vec![]);
        }
    };
    let model = config.llm_model.clone().unwrap_or(db_model);
    let base_url = config.openai_base_url.clone().or(db_base_url);

    let processing: Option<(i64,)> = sqlx::query_as(
        "SELECT num_segments_to_input_to_prompt FROM processing_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None);
    let chunk_size = processing.map(|(n,)| n as usize).unwrap_or(60);

    let config = ClassifierConfig {
        api_key,
        model,
        base_url,
        timeout_sec: timeout as u64,
        max_tokens: max_tokens as u32,
        max_concurrent: max_concurrent as u32,
        max_retries: max_retries as u32,
        chunk_size,
        min_confidence,
        enable_boundary_refinement: true,
    };

    classifier::classify_segments(pool, post_id, segments, &config, feed_title, feed_description)
        .await
        .map_err(|e| PipelineError::Classification(e.to_string()))
}

async fn classify_oneshot(
    pool: &SqlitePool,
    app_config: &AppConfig,
    post_id: i64,
    segments: &[Segment],
    feed_title: &str,
    feed_description: &str,
) -> Result<Vec<IdentifiedAd>, PipelineError> {
    // Check for existing identifications
    let existing_count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM identification i
         JOIN transcript_segment ts ON ts.id = i.transcript_segment_id
         WHERE ts.post_id = ?",
    )
    .bind(post_id)
    .fetch_one(pool)
    .await
    .unwrap_or((0,));

    if existing_count.0 > 0 {
        let existing: Vec<(i64, f64, f64, f64, String)> = sqlx::query_as(
            "SELECT ts.id, ts.start_time, ts.end_time, i.confidence, i.label
             FROM identification i
             JOIN transcript_segment ts ON ts.id = i.transcript_segment_id
             WHERE ts.post_id = ? AND i.label = 'ad'
             ORDER BY ts.start_time",
        )
        .bind(post_id)
        .fetch_all(pool)
        .await
        .unwrap_or_default();

        return Ok(existing
            .into_iter()
            .map(|(id, start, end, conf, label)| IdentifiedAd {
                segment_id: id,
                start_time: start,
                end_time: end,
                confidence: conf,
                label,
            })
            .collect());
    }

    // Load LLM settings
    let llm: Option<(Option<String>, String, Option<String>, Option<String>, i64, i64, i64, i64, i64, f64)> =
        sqlx::query_as(
            "SELECT l.llm_api_key, l.llm_model, l.oneshot_model, l.openai_base_url, l.openai_timeout, l.openai_max_tokens, l.llm_max_retry_attempts, l.oneshot_max_chunk_duration_seconds, l.oneshot_chunk_overlap_seconds, o.min_confidence
             FROM llm_settings l, output_settings o WHERE l.id = 1 AND o.id = 1",
        )
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

    let Some((db_api_key, db_llm_model, db_oneshot_model, db_base_url, timeout, max_tokens, max_retries, max_chunk_duration, chunk_overlap, min_confidence)) = llm else {
        tracing::warn!("LLM settings not configured — skipping one-shot classification");
        return Ok(vec![]);
    };

    // Env var precedence: ONESHOT_API_KEY > LLM_API_KEY > DB
    let api_key = app_config.oneshot_api_key.clone()
        .or_else(|| app_config.llm_api_key.clone())
        .or(db_api_key)
        .filter(|k| !k.is_empty());

    let Some(api_key) = api_key else {
        tracing::warn!("No LLM API key configured — skipping one-shot classification");
        return Ok(vec![]);
    };

    // Env var precedence: ONESHOT_MODEL > LLM_MODEL > DB oneshot_model > DB llm_model
    let model = app_config.oneshot_model.clone()
        .or_else(|| app_config.llm_model.clone())
        .unwrap_or_else(|| {
            db_oneshot_model
                .filter(|m| !m.is_empty())
                .unwrap_or(db_llm_model)
        });

    let base_url = app_config.openai_base_url.clone().or(db_base_url);

    let config = ClassifierConfig {
        api_key,
        model,
        base_url,
        timeout_sec: timeout as u64,
        max_tokens: max_tokens as u32,
        max_concurrent: 1,
        max_retries: max_retries as u32,
        chunk_size: 0, // not used by oneshot
        min_confidence,
        enable_boundary_refinement: false,
    };

    crate::classification::oneshot::classify_oneshot(
        pool,
        post_id,
        segments,
        &config,
        feed_title,
        feed_description,
        max_chunk_duration as f64,
        chunk_overlap as f64,
    )
    .await
    .map_err(|e| PipelineError::Classification(e.to_string()))
}

/// Chapter-based ad detection: read chapters from audio via ffprobe, filter by keywords.
async fn classify_chapters(
    audio_path: &Path,
    filter_strings_csv: Option<&str>,
) -> Result<Vec<(f64, f64)>, PipelineError> {
    let default_filters = "sponsor,ad,advertisement,promo,commercial";
    let filter_csv = filter_strings_csv.unwrap_or(default_filters);
    let filters: Vec<String> = filter_csv
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();

    // Read chapters via ffprobe
    let output = tokio::process::Command::new("ffprobe")
        .args([
            "-v", "quiet",
            "-print_format", "json",
            "-show_chapters",
        ])
        .arg(audio_path)
        .output()
        .await
        .map_err(|e| PipelineError::Io(format!("ffprobe failed: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(PipelineError::Io(format!("ffprobe error: {stderr}")));
    }

    let probe: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| PipelineError::Io(format!("ffprobe JSON parse error: {e}")))?;

    let chapters = probe.get("chapters").and_then(|c| c.as_array());
    let Some(chapters) = chapters else {
        tracing::warn!("No chapters found in audio — chapter strategy producing no ads");
        return Ok(vec![]);
    };

    if chapters.is_empty() {
        return Ok(vec![]);
    }

    let mut ad_segments = Vec::new();
    for ch in chapters {
        let title = ch
            .get("tags")
            .and_then(|t| t.get("title"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        let title_lower = title.to_lowercase();

        let is_ad = filters.iter().any(|f| title_lower.contains(f.as_str()));
        if is_ad {
            let start = ch
                .get("start_time")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);
            let end = ch
                .get("end_time")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0);
            if end > start {
                tracing::info!("Chapter ad: \"{title}\" ({start:.1}s - {end:.1}s)");
                ad_segments.push((start, end));
            }
        }
    }

    tracing::info!(
        "Chapter-based detection: {}/{} chapters matched as ads",
        ad_segments.len(),
        chapters.len()
    );

    // Also load global defaults from DB if no per-feed filter was set
    // (already handled via the fallback in filter_strings_csv)

    Ok(ad_segments)
}

async fn refine(
    pool: &SqlitePool,
    app_config: &AppConfig,
    post_id: i64,
    ad_segments: &[IdentifiedAd],
    transcript: &[Segment],
) -> Vec<(f64, f64)> {
    if ad_segments.is_empty() {
        return vec![];
    }

    let enabled: Option<(bool,)> = sqlx::query_as(
        "SELECT enable_boundary_refinement FROM llm_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    if !enabled.map(|(e,)| e).unwrap_or(true) {
        return ad_segments.iter().map(|a| (a.start_time, a.end_time)).collect();
    }

    let llm: Option<(Option<String>, String, Option<String>, i64)> = sqlx::query_as(
        "SELECT llm_api_key, llm_model, openai_base_url, openai_timeout FROM llm_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None);

    let Some((db_api_key, db_model, db_base_url, timeout)) = llm else {
        return ad_segments.iter().map(|a| (a.start_time, a.end_time)).collect();
    };

    // Env var precedence: env > DB
    let api_key = app_config.llm_api_key.clone().or(db_api_key);
    let Some(api_key) = api_key.filter(|k| !k.is_empty()) else {
        return ad_segments.iter().map(|a| (a.start_time, a.end_time)).collect();
    };
    let model = app_config.llm_model.clone().unwrap_or(db_model);
    let base_url = app_config.openai_base_url.clone().or(db_base_url);

    let config = ClassifierConfig {
        api_key,
        model,
        base_url,
        timeout_sec: timeout as u64,
        max_tokens: 500,
        max_concurrent: 1,
        max_retries: 2,
        chunk_size: 60,
        min_confidence: 0.0,
        enable_boundary_refinement: true,
    };

    let ads_with_confidence: Vec<(f64, f64, f64)> = ad_segments
        .iter()
        .map(|a| (a.start_time, a.end_time, a.confidence))
        .collect();

    crate::refinement::refine_boundaries(pool, post_id, &ads_with_confidence, transcript, &config).await
}

async fn cut_audio(
    input: &Path,
    ad_segments: &[(f64, f64)],
    feed_title: &str,
    post_title: &str,
) -> Result<PathBuf, PipelineError> {
    let sanitized_feed = sanitize_filename(feed_title);
    let sanitized_post = sanitize_filename(post_title);
    let output_dir = PathBuf::from("data/srv").join(&sanitized_feed);
    tokio::fs::create_dir_all(&output_dir)
        .await
        .map_err(|e| PipelineError::Io(e.to_string()))?;

    let output_path = output_dir.join(format!("{sanitized_post}.mp3"));

    if ad_segments.is_empty() {
        tokio::fs::copy(input, &output_path)
            .await
            .map_err(|e| PipelineError::Io(e.to_string()))?;
    } else {
        crate::audio::clip_segments_with_fade(
            input.to_str().unwrap_or(""),
            output_path.to_str().unwrap_or(""),
            &ad_segments
                .iter()
                .map(|(s, e)| ((*s * 1000.0) as i64, (*e * 1000.0) as i64))
                .collect::<Vec<_>>(),
            50,
            false,
        )
        .await
        .map_err(|e| PipelineError::Audio(e.to_string()))?;
    }

    Ok(output_path)
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim()
        .to_string()
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("validation: {0}")]
    Validation(String),
    #[error("download error: {0}")]
    Download(String),
    #[error("transcription error: {0}")]
    Transcription(String),
    #[error("classification error: {0}")]
    Classification(String),
    #[error("audio error: {0}")]
    Audio(String),
    #[error("database error: {0}")]
    Db(String),
    #[error("io error: {0}")]
    Io(String),
}
