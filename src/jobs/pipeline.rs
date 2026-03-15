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
    let post: Option<(i64, i64, String, Option<String>, Option<String>, Option<String>, Option<String>)> =
        sqlx::query_as(
            "SELECT id, feed_id, title, download_url, unprocessed_audio_path, processed_audio_path, description FROM post WHERE guid = ?",
        )
        .bind(post_guid)
        .fetch_optional(pool)
        .await
        .map_err(|e| PipelineError::Db(e.to_string()))?;

    let Some((post_id, feed_id, post_title, download_url, existing_audio, processed_audio, post_description)) = post
    else {
        return Err(PipelineError::NotFound("Post not found".into()));
    };

    if let Some(processed) = &processed_audio {
        if Path::new(processed).exists() {
            update_step(pool, job_id, 4, "Processing complete", 100.0).await;
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
    let _feed_description = feed_description.unwrap_or_else(|| feed_title.clone());

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

    // Step 1: Download (matches Python step 1 "Downloading episode")
    update_step(pool, job_id, 1, "Downloading episode", 0.0).await;
    let audio_path = if let Some(existing) = &existing_audio {
        if Path::new(existing).exists() {
            PathBuf::from(existing)
        } else {
            download_audio(pool, post_id, &download_url, &post_title, job_id).await?
        }
    } else {
        download_audio(pool, post_id, &download_url, &post_title, job_id).await?
    };
    update_step(pool, job_id, 1, "Downloading episode", 25.0).await;

    // Strategy-dependent steps
    let refined = match strategy.as_str() {
        "chapter" => {
            // Chapter-based: skip transcription + LLM, read chapters from audio
            update_step(pool, job_id, 2, "Reading chapters", 30.0).await;
            let chapter_ads = classify_chapters(&audio_path, chapter_filter_strings.as_deref()).await?;
            update_step(pool, job_id, 2, "Reading chapters", 50.0).await;
            update_step(pool, job_id, 3, "Filtering chapters", 70.0).await;
            chapter_ads
        }
        _ => {
            // LLM-based or oneshot: transcribe, classify, refine
            // Step 2: Transcribe (matches Python step 2 "Transcribing audio")
            update_step(pool, job_id, 2, "Transcribing audio", 30.0).await;
            let segments = transcribe(pool, config, post_id, &audio_path).await?;
            update_step(pool, job_id, 2, "Transcribing audio", 50.0).await;

            // Step 3: Classify (matches Python step 3 "Classifying ads")
            let step_name = if strategy.as_str() == "oneshot" {
                "Classifying ads (one-shot)"
            } else {
                "Classifying ads"
            };
            update_step(pool, job_id, 3, step_name, 55.0).await;
            // Python parity: classifiers use post title/description, not feed-level
            let classify_title = &post_title;
            let classify_desc = post_description.as_deref().unwrap_or("");
            let ad_segments = match strategy.as_str() {
                "oneshot" => classify_oneshot(pool, config, post_id, &segments, classify_title, classify_desc).await?,
                _ => classify(pool, config, post_id, &segments, classify_title, classify_desc).await?,
            };
            update_step(pool, job_id, 3, step_name, 70.0).await;

            // Apply boundary refinement to all strategies (Python parity)
            refine(pool, config, post_id, &ad_segments, &segments).await
        }
    };

    // Step 4: Process audio (matches Python step 4 "Processing audio")
    update_step(pool, job_id, 4, "Processing audio", 75.0).await;

    // Read output settings (fade_ms, min_ad_segment_separation, min_ad_segment_length)
    let (fade_ms, min_sep, min_len): (i64, i64, i64) = sqlx::query_as::<_, (i64, i64, i64)>(
        "SELECT fade_ms, min_ad_segement_separation_seconds, min_ad_segment_length_seconds FROM output_settings WHERE id = 1",
    )
    .fetch_optional(pool)
    .await
    .unwrap_or(None)
    .unwrap_or((3000, 8, 3));

    // Get audio duration for last-segment extension
    let audio_duration = crate::audio::get_audio_duration_ms(audio_path.to_str().unwrap_or(""))
        .await
        .unwrap_or(0) as f64 / 1000.0;

    // Merge ad segments (proximity + content-aware keyword merge, Python parity: AdMerger)
    let merged = merge_ad_segments(pool, post_id, &refined, min_sep as f64, min_len as f64, audio_duration).await;
    tracing::info!("Merged {} ad segments into {} groups (gap: {}s, min_len: {}s)", refined.len(), merged.len(), min_sep, min_len);

    let output_path = cut_audio(&audio_path, &merged, &feed_title, &post_title, fade_ms).await?;
    update_step(pool, job_id, 4, "Processing audio", 90.0).await;

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

    update_step(pool, job_id, 4, "Processing complete", 100.0).await;
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

    // Python parity: User-Agent header, Referer for acast, 60s timeout
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| PipelineError::Download(e.to_string()))?;
    let mut req = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36");
    if url.contains("acast.com") {
        req = req.header("Referer", "https://www.acast.com/");
    }
    let mut response = req.send()
        .await
        .map_err(|e| PipelineError::Download(e.to_string()))?;

    if !response.status().is_success() {
        return Err(PipelineError::Download(format!(
            "HTTP {} for {url}",
            response.status()
        )));
    }

    // Stream to file instead of loading entirely into memory (Python parity)
    let mut file = tokio::fs::File::create(&path)
        .await
        .map_err(|e| PipelineError::Io(e.to_string()))?;
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = response.chunk().await.map_err(|e| PipelineError::Download(e.to_string()))? {
        file.write_all(&chunk)
            .await
            .map_err(|e| PipelineError::Io(e.to_string()))?;
    }
    file.flush().await.map_err(|e| PipelineError::Io(e.to_string()))?;

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
    let llm: Option<(Option<String>, String, Option<String>, i64, i64, i64, i64, f64, Option<i64>)> =
        sqlx::query_as(
            "SELECT l.llm_api_key, l.llm_model, l.openai_base_url, l.openai_timeout, l.openai_max_tokens, l.llm_max_concurrent_calls, l.llm_max_retry_attempts, o.min_confidence, l.llm_max_input_tokens_per_call
             FROM llm_settings l, output_settings o WHERE l.id = 1 AND o.id = 1",
        )
        .fetch_optional(pool)
        .await
        .unwrap_or(None);

    let Some((db_api_key, db_model, db_base_url, timeout, max_tokens, max_concurrent, max_retries, min_confidence, max_input_tokens)) = llm else {
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
        max_input_tokens_per_call: max_input_tokens.map(|v| v as u32),
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
        max_input_tokens_per_call: None, // oneshot handles its own chunking
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
        max_tokens: 4096,
        max_concurrent: 1,
        max_retries: 2,
        chunk_size: 60,
        min_confidence: 0.0,
        enable_boundary_refinement: true,
        max_input_tokens_per_call: None, // refinement prompts are small
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
    fade_ms: i64,
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
            fade_ms,
            false,
        )
        .await
        .map_err(|e| PipelineError::Audio(e.to_string()))?;
    }

    Ok(output_path)
}

/// Merge ad segments: proximity grouping, content-aware keyword merge,
/// short-segment filtering, last-segment extension.
/// Matches Python's AdMerger + AudioProcessor.merge_ad_segments.
async fn merge_ad_segments(
    pool: &SqlitePool,
    post_id: i64,
    segments: &[(f64, f64)],
    max_gap: f64,
    min_segment_length: f64,
    audio_duration: f64,
) -> Vec<(f64, f64)> {
    if segments.is_empty() {
        return vec![];
    }

    let mut sorted: Vec<(f64, f64)> = segments.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Save last segment before filtering (for restoration)
    let last_segment = *sorted.last().unwrap();
    let last_near_end = audio_duration > 0.0 && (audio_duration - last_segment.1) <= max_gap;

    // Pass 1: Proximity merge
    let mut merged = proximity_merge(&sorted, max_gap);

    // Pass 2: Content-aware merge — merge groups with shared keywords
    if merged.len() > 1 {
        merged = content_aware_merge(pool, post_id, &merged, max_gap * 1.5).await;
    }

    // Pass 3: Filter short segments
    merged.retain(|&(start, end)| (end - start) >= min_segment_length);

    // Pass 4: Restore last segment if it was near the end and got filtered
    if last_near_end && (merged.is_empty() || merged.last().unwrap().1 < last_segment.0) {
        merged.push(last_segment);
    }

    // Pass 5: Re-merge after filtering
    if merged.len() > 1 {
        merged = proximity_merge(&merged, max_gap);
    }

    // Pass 6: Extend last segment to audio end if close
    if audio_duration > 0.0 {
        if let Some(last) = merged.last_mut() {
            if (audio_duration - last.1) <= max_gap {
                last.1 = audio_duration;
            }
        }
    }

    merged
}

/// Content-aware merge: query transcript text for each group, extract keywords
/// (URLs, promo codes, phone numbers, repeated brand names), merge groups
/// that share keywords or both have high confidence.
/// Matches Python's AdMerger._refine_by_content + _should_merge + _extract_keywords.
async fn content_aware_merge(
    pool: &SqlitePool,
    post_id: i64,
    groups: &[(f64, f64)],
    min_content_gap: f64,
) -> Vec<(f64, f64)> {
    // Extract keywords for each group from transcript text
    let mut group_keywords: Vec<Vec<String>> = Vec::new();
    for &(start, end) in groups {
        let keywords = extract_keywords_for_range(pool, post_id, start, end).await;
        group_keywords.push(keywords);
    }

    let mut merged: Vec<(f64, f64)> = Vec::new();
    let mut merged_kw: Vec<Vec<String>> = Vec::new();
    let mut i = 0;

    while i < groups.len() {
        let current = groups[i];
        let current_kw = &group_keywords[i];

        if i + 1 < groups.len() {
            let next = groups[i + 1];
            let next_kw = &group_keywords[i + 1];
            let gap = next.0 - current.1;

            if gap <= min_content_gap && should_merge_groups(current_kw, next_kw) {
                // Merge the two groups
                let combined_kw: Vec<String> = current_kw.iter()
                    .chain(next_kw.iter())
                    .cloned()
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                merged.push((current.0, next.1));
                merged_kw.push(combined_kw);
                i += 2;
                continue;
            }
        }

        merged.push(current);
        merged_kw.push(current_kw.clone());
        i += 1;
    }

    // Filter weak groups: long segments (>180s) without keywords are likely
    // educational/self-promo, not ads (Python's _is_valid_group)
    merged.into_iter()
        .zip(merged_kw.iter())
        .filter(|&((start, end), kw)| {
            let duration = end - start;
            if duration > 180.0 && kw.is_empty() {
                tracing::debug!("Filtering weak group {:.1}s-{:.1}s (long, no keywords)", start, end);
                false
            } else {
                true
            }
        })
        .map(|(seg, _)| seg)
        .collect()
}

/// Check if two ad groups should be merged based on shared keywords.
/// Matches Python's AdMerger._should_merge.
fn should_merge_groups(kw1: &[String], kw2: &[String]) -> bool {
    // Shared keywords → merge
    let set1: std::collections::HashSet<&str> = kw1.iter().map(|s| s.as_str()).collect();
    for k in kw2 {
        if set1.contains(k.as_str()) {
            return true;
        }
    }
    false
}

/// Extract keywords (URLs, promo codes, phone numbers, brand names) from
/// transcript text in a time range. Matches Python's AdMerger._extract_keywords.
async fn extract_keywords_for_range(
    pool: &SqlitePool,
    post_id: i64,
    start: f64,
    end: f64,
) -> Vec<String> {
    // Query transcript text for this time range
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT text FROM transcript_segment WHERE post_id = ? AND start_time >= ? AND end_time <= ? ORDER BY start_time",
    )
    .bind(post_id)
    .bind(start)
    .bind(end)
    .fetch_all(pool)
    .await
    .unwrap_or_default();

    let text: String = rows.iter().map(|(t,)| t.as_str()).collect::<Vec<_>>().join(" ");
    let text_lower = text.to_lowercase();

    let mut keywords: Vec<String> = Vec::new();

    // URLs: *.com, *.net, *.org, *.io
    let url_re = regex::Regex::new(r"\b([a-z0-9\-\.]+\.(?:com|net|org|io))\b").unwrap();
    for cap in url_re.captures_iter(&text_lower) {
        keywords.push(cap[1].to_string());
    }

    // Promo codes: "code X", "promo X", "save X"
    let promo_re = regex::Regex::new(r"(?i)\b(code|promo|save)\s+\w+\b").unwrap();
    for cap in promo_re.captures_iter(&text_lower) {
        keywords.push(cap[0].to_string());
    }

    // Phone numbers
    let phone_re = regex::Regex::new(r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b").unwrap();
    if phone_re.is_match(&text) {
        keywords.push("phone".to_string());
    }

    // Brand names: capitalized words appearing 2+ times
    let brand_re = regex::Regex::new(r"\b[A-Z][a-z]+\b").unwrap();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in brand_re.find_iter(&text) {
        let word = m.as_str().to_string();
        if word.len() > 3 {
            *counts.entry(word).or_insert(0) += 1;
        }
    }
    for (word, count) in &counts {
        if *count >= 2 {
            keywords.push(word.to_lowercase());
        }
    }

    keywords.sort();
    keywords.dedup();
    keywords
}

fn proximity_merge(segments: &[(f64, f64)], max_gap: f64) -> Vec<(f64, f64)> {
    let mut merged: Vec<(f64, f64)> = Vec::new();
    let mut current = segments[0];
    for &(start, end) in &segments[1..] {
        if start - current.1 <= max_gap {
            current.1 = current.1.max(end);
        } else {
            merged.push(current);
            current = (start, end);
        }
    }
    merged.push(current);
    merged
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
