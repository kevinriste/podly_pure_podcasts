use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use super::cue_detector::CueDetector;

/// System prompt for ad classification.
const SYSTEM_PROMPT: &str = include_str!("../../prompts/system_prompt.txt");

/// Transcript segment from the database.
#[derive(Debug, Clone)]
pub struct Segment {
    pub id: i64,
    pub sequence_num: i64,
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
}

/// LLM response for a chunk of transcript.
#[derive(Debug, Deserialize, Serialize)]
pub struct LlmClassificationResponse {
    pub ad_segments: Vec<LlmAdPrediction>,
    pub content_type: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LlmAdPrediction {
    pub segment_offset: f64,
    pub confidence: f64,
}

/// Configuration for the classifier.
#[allow(dead_code)]
pub struct ClassifierConfig {
    pub api_key: String,
    pub model: String,
    pub base_url: Option<String>,
    pub timeout_sec: u64,
    pub max_tokens: u32,
    pub max_concurrent: u32,
    pub max_retries: u32,
    pub chunk_size: usize,
    pub min_confidence: f64,
    pub enable_boundary_refinement: bool,
}

/// Result of classification: identified ad segments.
#[derive(Debug, Clone, Serialize)]
pub struct IdentifiedAd {
    pub segment_id: i64,
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f64,
    pub label: String,
}

/// Run ad classification on transcript segments for a post.
///
/// 1. Chunk segments into overlapping windows
/// 2. For each chunk, build a prompt and call the LLM
/// 3. Parse responses, map offsets to segments
/// 4. Create identifications in the database
/// 5. Expand neighbors using cue detection heuristics
pub async fn classify_segments(
    pool: &SqlitePool,
    post_id: i64,
    segments: &[Segment],
    config: &ClassifierConfig,
    feed_title: &str,
    feed_description: &str,
) -> Result<Vec<IdentifiedAd>, ClassificationError> {
    if segments.is_empty() {
        return Ok(vec![]);
    }

    let cue_detector = CueDetector::new();
    let mut all_identified: Vec<IdentifiedAd> = Vec::new();

    // Build chunks with overlap
    let chunk_size = config.chunk_size;
    let overlap = chunk_size / 4; // 25% overlap
    let mut chunk_start = 0;

    let semaphore = tokio::sync::Semaphore::new(config.max_concurrent as usize);

    while chunk_start < segments.len() {
        let chunk_end = (chunk_start + chunk_size).min(segments.len());
        let chunk = &segments[chunk_start..chunk_end];
        let is_start = chunk_start == 0;
        let is_end = chunk_end == segments.len();

        // Build user prompt
        let user_prompt = build_user_prompt(
            chunk,
            &cue_detector,
            feed_title,
            feed_description,
            is_start,
            is_end,
        );

        // Call LLM with retries
        let _permit = semaphore.acquire().await.map_err(|_| {
            ClassificationError::Internal("semaphore closed".into())
        })?;

        let response = call_llm_with_retries(config, &user_prompt, config.max_retries).await;

        match response {
            Ok(llm_response) => {
                // Record model call in DB
                let first_seq = chunk.first().map(|s| s.sequence_num).unwrap_or(0);
                let last_seq = chunk.last().map(|s| s.sequence_num).unwrap_or(0);
                let model_call_id = record_model_call(
                    pool,
                    post_id,
                    first_seq,
                    last_seq,
                    &config.model,
                    &user_prompt,
                    &serde_json::to_string(&llm_response).unwrap_or_default(),
                    "success",
                )
                .await;

                // Map predictions to actual segments
                let identified =
                    map_predictions_to_segments(chunk, &llm_response, model_call_id, config.min_confidence);
                all_identified.extend(identified);
            }
            Err(e) => {
                let first_seq = chunk.first().map(|s| s.sequence_num).unwrap_or(0);
                let last_seq = chunk.last().map(|s| s.sequence_num).unwrap_or(0);
                let _model_call_id = record_model_call(
                    pool,
                    post_id,
                    first_seq,
                    last_seq,
                    &config.model,
                    &user_prompt,
                    "",
                    "failed_permanent",
                )
                .await;
                tracing::warn!("LLM classification failed for chunk {chunk_start}..{chunk_end}: {e}");
            }
        }

        // Advance by chunk_size minus overlap
        if chunk_end >= segments.len() {
            break;
        }
        chunk_start = chunk_end.saturating_sub(overlap);
    }

    // Deduplicate by segment_id (keep highest confidence)
    all_identified.sort_by(|a, b| a.segment_id.cmp(&b.segment_id));
    all_identified.dedup_by(|a, b| {
        if a.segment_id == b.segment_id {
            // Keep the one with higher confidence (in b since dedup keeps b)
            if a.confidence > b.confidence {
                b.confidence = a.confidence;
            }
            true
        } else {
            false
        }
    });

    // Expand neighbors using cue detection
    let expanded = expand_neighbors(&all_identified, segments, &cue_detector, config.min_confidence);
    all_identified.extend(expanded);
    all_identified.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap_or(std::cmp::Ordering::Equal));
    all_identified.dedup_by(|a, b| a.segment_id == b.segment_id);

    // Persist identifications to DB
    for ad in &all_identified {
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO identification (transcript_segment_id, model_call_id, confidence, label) VALUES (?, 0, ?, ?)",
        )
        .bind(ad.segment_id)
        .bind(ad.confidence)
        .bind(&ad.label)
        .execute(pool)
        .await;
    }

    Ok(all_identified)
}

fn build_user_prompt(
    chunk: &[Segment],
    cue_detector: &CueDetector,
    feed_title: &str,
    feed_description: &str,
    is_start: bool,
    is_end: bool,
) -> String {
    let mut transcript = String::new();

    if is_start {
        transcript.push_str("=== TRANSCRIPT START ===\n");
    }

    for seg in chunk {
        let highlighted = cue_detector.highlight_cues(&seg.text);
        transcript.push_str(&format!("[{}] {}\n", seg.start_time, highlighted));
    }

    if is_end {
        transcript.push_str("=== TRANSCRIPT END ===\n");
    }

    format!(
        "You are analyzing \"{feed_title}\", a podcast about {feed_description}.\n\
         Return only the JSON contract described in the system prompt using the transcript excerpt below.\n\n\
         {transcript}"
    )
}

async fn call_llm_with_retries(
    config: &ClassifierConfig,
    user_prompt: &str,
    max_retries: u32,
) -> Result<LlmClassificationResponse, ClassificationError> {
    let genai_client = crate::llm::build_genai_client(
        &config.api_key,
        &config.model,
        config.base_url.as_deref(),
    )
    .map_err(|e| ClassificationError::Api(e.to_string()))?;

    for attempt in 0..=max_retries {
        let result = crate::llm::chat_completion(
            &genai_client,
            &config.model,
            Some(SYSTEM_PROMPT),
            user_prompt,
            Some(0.1),
            Some(config.max_tokens as u32),
        )
        .await;

        match result {
            Ok(content) => {
                return parse_llm_response(&content);
            }
            Err(e) => {
                let err_str = e.to_string();
                let is_rate_limit = err_str.contains("429") || err_str.contains("rate");
                if is_rate_limit {
                    let wait = std::time::Duration::from_secs(60 * 2u64.pow(attempt));
                    tracing::warn!(
                        "LLM rate limited (attempt {}/{}), waiting {}s",
                        attempt + 1,
                        max_retries + 1,
                        wait.as_secs()
                    );
                    tokio::time::sleep(wait).await;
                } else if attempt < max_retries {
                    let wait = std::time::Duration::from_secs(2u64.pow(attempt));
                    tracing::warn!("LLM error (attempt {}): {e}, retrying in {}s", attempt + 1, wait.as_secs());
                    tokio::time::sleep(wait).await;
                } else {
                    return Err(ClassificationError::Api(err_str));
                }
            }
        }
    }

    Err(ClassificationError::Api("max retries exceeded".into()))
}

/// Parse and clean LLM response JSON, with repair for truncated output.
fn parse_llm_response(raw: &str) -> Result<LlmClassificationResponse, ClassificationError> {
    // Find first { in response (skip markdown fences, etc)
    let start = raw.find('{').ok_or_else(|| {
        ClassificationError::Parse(format!("no JSON object in response: {}", &raw[..raw.len().min(200)]))
    })?;
    let mut json_str = &raw[start..];

    // Trim to last } if brackets are balanced
    let open = json_str.matches('{').count();
    let close = json_str.matches('}').count();
    if close >= open {
        if let Some(end) = json_str.rfind('}') {
            json_str = &json_str[..=end];
        }
    }

    let cleaned = json_str.replace('\'', "\"").replace('\n', "");

    // First attempt
    if let Ok(parsed) = serde_json::from_str::<LlmClassificationResponse>(&cleaned) {
        return Ok(parsed);
    }

    // Attempt repair: add missing brackets
    let mut repaired = cleaned.trim_end().trim_end_matches(',').to_string();
    let open_braces = repaired.matches('{').count();
    let close_braces = repaired.matches('}').count();
    let open_brackets = repaired.matches('[').count();
    let close_brackets = repaired.matches(']').count();

    for _ in 0..(open_brackets.saturating_sub(close_brackets)) {
        repaired.push(']');
    }
    for _ in 0..(open_braces.saturating_sub(close_braces)) {
        repaired.push('}');
    }

    serde_json::from_str::<LlmClassificationResponse>(&repaired).map_err(|e| {
        ClassificationError::Parse(format!("failed to parse LLM response: {e}"))
    })
}

fn map_predictions_to_segments(
    chunk: &[Segment],
    response: &LlmClassificationResponse,
    model_call_id: i64,
    min_confidence: f64,
) -> Vec<IdentifiedAd> {
    let content_type = response.content_type.as_deref().unwrap_or("unknown");

    // Confidence penalty based on content type
    let confidence_penalty = match content_type {
        "educational/self_promo" => 0.25,
        "technical_discussion" => 0.25,
        "transition" => 0.1,
        _ => 0.0,
    };

    let mut identified = Vec::new();

    for pred in &response.ad_segments {
        let adjusted_confidence = (pred.confidence - confidence_penalty).max(0.0);
        if adjusted_confidence < min_confidence {
            continue;
        }

        // Find closest segment by start_time (within 0.5s tolerance)
        let best_match = chunk.iter().min_by(|a, b| {
            let da = (a.start_time - pred.segment_offset).abs();
            let db = (b.start_time - pred.segment_offset).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });

        if let Some(seg) = best_match {
            if (seg.start_time - pred.segment_offset).abs() <= 0.5 {
                identified.push(IdentifiedAd {
                    segment_id: seg.id,
                    start_time: seg.start_time,
                    end_time: seg.end_time,
                    confidence: adjusted_confidence,
                    label: "ad".to_string(),
                });
            }
        }
    }

    let _ = model_call_id; // Used for DB tracking
    identified
}

/// Expand to neighboring segments using cue detection heuristics.
fn expand_neighbors(
    identified: &[IdentifiedAd],
    all_segments: &[Segment],
    cue_detector: &CueDetector,
    min_confidence: f64,
) -> Vec<IdentifiedAd> {
    let ad_ids: std::collections::HashSet<i64> = identified.iter().map(|a| a.segment_id).collect();
    let mut expanded = Vec::new();

    for ad in identified {
        // Look at neighboring segments
        for seg in all_segments {
            if ad_ids.contains(&seg.id) {
                continue;
            }

            let gap = (seg.start_time - ad.end_time).abs().min((ad.start_time - seg.end_time).abs());
            if gap > 10.0 {
                continue;
            }

            let cues = cue_detector.analyze(&seg.text);

            let confidence = if cues.transition {
                0.72
            } else if cues.has_strong_cue() {
                if gap <= 10.0 { 0.85 } else { 0.8 }
            } else {
                continue;
            };

            let adjusted = if cues.self_promo {
                confidence - 0.25
            } else {
                confidence
            };

            if adjusted >= min_confidence {
                expanded.push(IdentifiedAd {
                    segment_id: seg.id,
                    start_time: seg.start_time,
                    end_time: seg.end_time,
                    confidence: adjusted,
                    label: "ad".to_string(),
                });
            }
        }
    }

    expanded
}

#[allow(clippy::too_many_arguments)]
async fn record_model_call(
    pool: &SqlitePool,
    post_id: i64,
    first_seq: i64,
    last_seq: i64,
    model_name: &str,
    prompt: &str,
    response: &str,
    status: &str,
) -> i64 {
    let now = chrono::Utc::now().to_rfc3339();
    let result = sqlx::query(
        "INSERT INTO model_call (post_id, first_segment_sequence_num, last_segment_sequence_num, model_name, prompt, response, timestamp, status)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(post_id, first_segment_sequence_num, last_segment_sequence_num, model_name)
         DO UPDATE SET response = excluded.response, status = excluded.status",
    )
    .bind(post_id)
    .bind(first_seq)
    .bind(last_seq)
    .bind(model_name)
    .bind(prompt)
    .bind(response)
    .bind(&now)
    .bind(status)
    .execute(pool)
    .await;

    match result {
        Ok(r) => r.last_insert_rowid(),
        Err(e) => {
            tracing::warn!("Failed to record model call: {e}");
            0
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClassificationError {
    #[error("api error: {0}")]
    Api(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("internal: {0}")]
    Internal(String),
}
