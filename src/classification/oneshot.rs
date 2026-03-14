use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use super::classifier::{ClassifierConfig, IdentifiedAd, Segment};

/// System prompt for one-shot ad classification.
const ONESHOT_SYSTEM_PROMPT: &str = include_str!("../../prompts/oneshot_system_prompt.txt");

/// One-shot ad segment from the LLM response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OneShotAdSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub confidence: f64,
    pub ad_type: Option<String>,
    pub reason: Option<String>,
}

/// LLM response envelope.
#[derive(Debug, Deserialize)]
struct OneShotResponse {
    ad_segments: Vec<OneShotAdSegment>,
}

/// Run one-shot classification on transcript segments.
///
/// For episodes ≤ `max_chunk_duration` seconds, sends the entire transcript in one LLM call.
/// For longer episodes, splits into chunks with overlap and deduplicates.
pub async fn classify_oneshot(
    pool: &SqlitePool,
    post_id: i64,
    segments: &[Segment],
    config: &ClassifierConfig,
    feed_title: &str,
    feed_description: &str,
    max_chunk_duration: f64,
    chunk_overlap: f64,
) -> Result<Vec<IdentifiedAd>, OneShotError> {
    if segments.is_empty() {
        return Ok(vec![]);
    }

    let total_duration = segments.last().map(|s| s.end_time).unwrap_or(0.0);

    // Determine chunks
    let chunks = if total_duration <= max_chunk_duration {
        vec![(0, segments.len(), None::<String>)]
    } else {
        build_chunks(segments, max_chunk_duration, chunk_overlap, total_duration)
    };

    let mut all_ad_segments: Vec<OneShotAdSegment> = Vec::new();

    for (chunk_idx, (start_idx, end_idx, position_note)) in chunks.iter().enumerate() {
        let chunk_segments = &segments[*start_idx..*end_idx];
        if chunk_segments.is_empty() {
            continue;
        }

        // Build CSV transcript
        let transcript = build_csv(chunk_segments);

        // Build user prompt
        let position_note_str = position_note.as_deref().unwrap_or("");
        let user_prompt = format!(
            r#"Podcast: "{feed_title}"
Description: {feed_description}
Duration: {total_duration:.1} seconds
{position_note_str}

TRANSCRIPT CSV (columns: start_time,end_time,text):
{transcript}

Find all ad segments and a few adjacent transition segments on each side of the ad, if possible.
Err on the side of returning multiple segments with different confidence levels, especially where your confidence in something being an ad shifts.
Strongly prefer transition-aware segmentation with confidence gradients near ad boundaries (separate lower-confidence edge segments where appropriate).
I would rather have too much to work with, lots of low-confidence segments, than have no information at all about segments in and around an ad.
Return JSON."#
        );

        // Create model_call record
        let model_name = format!("oneshot:{}", config.model);
        let first_seq = chunk_segments.first().map(|s| s.sequence_num).unwrap_or(0);
        let last_seq = chunk_segments.last().map(|s| s.sequence_num).unwrap_or(0);
        let now = chrono::Utc::now().to_rfc3339();

        let mc_result = sqlx::query(
            "INSERT OR REPLACE INTO model_call (post_id, first_segment_sequence_num, last_segment_sequence_num, model_name, prompt, timestamp, status, retry_attempts) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0)",
        )
        .bind(post_id)
        .bind(first_seq)
        .bind(last_seq)
        .bind(&model_name)
        .bind(&user_prompt)
        .bind(&now)
        .execute(pool)
        .await
        .map_err(|e| OneShotError::Db(e.to_string()))?;

        let model_call_id = mc_result.last_insert_rowid();

        // Call LLM
        tracing::info!(
            "One-shot classification chunk {}/{} for post {post_id} ({} segments)",
            chunk_idx + 1,
            chunks.len(),
            chunk_segments.len()
        );

        let response = call_oneshot_llm(config, &user_prompt).await;

        match response {
            Ok(resp_text) => {
                // Update model_call with response
                let _ = sqlx::query(
                    "UPDATE model_call SET status = 'completed', response = ? WHERE id = ?",
                )
                .bind(&resp_text)
                .bind(model_call_id)
                .execute(pool)
                .await;

                // Parse response
                match parse_oneshot_response(&resp_text) {
                    Ok(parsed) => {
                        // Create identifications for each segment
                        for ad_seg in &parsed {
                            // Find overlapping transcript segments
                            for seg in chunk_segments {
                                let overlap_start = seg.start_time.max(ad_seg.start_time);
                                let overlap_end = seg.end_time.min(ad_seg.end_time);
                                if overlap_end > overlap_start {
                                    let label = if ad_seg.confidence >= config.min_confidence {
                                        "ad"
                                    } else {
                                        "ad_candidate"
                                    };

                                    let _ = sqlx::query(
                                        "INSERT OR REPLACE INTO identification (transcript_segment_id, model_call_id, confidence, label) VALUES (?, ?, ?, ?)",
                                    )
                                    .bind(seg.id)
                                    .bind(model_call_id)
                                    .bind(ad_seg.confidence)
                                    .bind(label)
                                    .execute(pool)
                                    .await;
                                }
                            }
                        }
                        all_ad_segments.extend(parsed);
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse one-shot response: {e}");
                        let _ = sqlx::query(
                            "UPDATE model_call SET status = 'error', error_message = ? WHERE id = ?",
                        )
                        .bind(format!("Parse error: {e}"))
                        .bind(model_call_id)
                        .execute(pool)
                        .await;
                    }
                }
            }
            Err(e) => {
                tracing::error!("One-shot LLM call failed: {e}");
                let _ = sqlx::query(
                    "UPDATE model_call SET status = 'error', error_message = ? WHERE id = ?",
                )
                .bind(e.to_string())
                .bind(model_call_id)
                .execute(pool)
                .await;
                return Err(OneShotError::Llm(e.to_string()));
            }
        }
    }

    // Deduplicate overlapping segments from chunk boundaries (keep higher confidence)
    let deduped = deduplicate_segments(all_ad_segments);

    // Convert to IdentifiedAd (using timestamp-based segment matching)
    let mut identified = Vec::new();
    for ad_seg in &deduped {
        // Find the best matching transcript segment
        let mut best_match: Option<&Segment> = None;
        let mut best_overlap = 0.0f64;
        for seg in segments {
            let overlap_start = seg.start_time.max(ad_seg.start_time);
            let overlap_end = seg.end_time.min(ad_seg.end_time);
            let overlap = (overlap_end - overlap_start).max(0.0);
            if overlap > best_overlap {
                best_overlap = overlap;
                best_match = Some(seg);
            }
        }

        if let Some(seg) = best_match {
            if ad_seg.confidence >= config.min_confidence {
                identified.push(IdentifiedAd {
                    segment_id: seg.id,
                    start_time: ad_seg.start_time,
                    end_time: ad_seg.end_time,
                    confidence: ad_seg.confidence,
                    label: "ad".into(),
                });
            }
        }
    }

    tracing::info!(
        "One-shot classification found {} ad segments ({} above threshold) for post {post_id}",
        deduped.len(),
        identified.len()
    );

    Ok(identified)
}

fn build_csv(segments: &[Segment]) -> String {
    let mut csv = String::from("start_time,end_time,text\n");
    for seg in segments {
        // Escape text for CSV (replace double quotes, wrap in quotes)
        let text = seg.text.replace('"', "\"\"");
        csv.push_str(&format!("{},{},\"{}\"\n", seg.start_time, seg.end_time, text));
    }
    csv
}

fn build_chunks(
    segments: &[Segment],
    max_duration: f64,
    overlap: f64,
    total_duration: f64,
) -> Vec<(usize, usize, Option<String>)> {
    let mut chunks = Vec::new();
    let mut chunk_start_time = 0.0f64;
    let mut chunk_num = 0usize;

    let total_chunks = ((total_duration - overlap) / (max_duration - overlap)).ceil() as usize;

    while chunk_start_time < total_duration {
        let chunk_end_time = (chunk_start_time + max_duration).min(total_duration);
        chunk_num += 1;

        // Find segment indices for this time range
        let start_idx = segments.iter().position(|s| s.end_time > chunk_start_time)
            .unwrap_or(segments.len());
        let end_idx = segments.iter().rposition(|s| s.start_time < chunk_end_time)
            .map(|i| i + 1)
            .unwrap_or(start_idx);

        let note = format!(
            "[This is chunk {chunk_num} of {total_chunks}, covering {chunk_start_time:.0}s to {chunk_end_time:.0}s of {total_duration:.0}s total]"
        );

        chunks.push((start_idx, end_idx, Some(note)));

        chunk_start_time = chunk_end_time - overlap;
        if chunk_end_time >= total_duration {
            break;
        }
    }

    chunks
}

fn deduplicate_segments(segments: Vec<OneShotAdSegment>) -> Vec<OneShotAdSegment> {
    if segments.is_empty() {
        return segments;
    }

    let mut sorted = segments;
    sorted.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    let mut deduped: Vec<OneShotAdSegment> = Vec::new();
    for seg in sorted {
        // Check if this overlaps with the last segment
        if let Some(last) = deduped.last_mut() {
            let overlap_start = last.start_time.max(seg.start_time);
            let overlap_end = last.end_time.min(seg.end_time);
            if overlap_end > overlap_start {
                // Overlapping — keep the higher confidence one, or merge
                if seg.confidence > last.confidence {
                    *last = seg;
                }
                continue;
            }
        }
        deduped.push(seg);
    }

    deduped
}

async fn call_oneshot_llm(
    config: &ClassifierConfig,
    user_prompt: &str,
) -> Result<String, OneShotError> {
    use genai::chat::{ChatMessage, ChatOptions, ChatRequest, ChatResponseFormat};

    let genai_client = crate::llm::build_genai_client(
        &config.api_key,
        &config.model,
        config.base_url.as_deref(),
    )
    .map_err(|e| OneShotError::Llm(e.to_string()))?;

    let messages = vec![
        ChatMessage::system(ONESHOT_SYSTEM_PROMPT),
        ChatMessage::user(user_prompt),
    ];
    let chat_req = ChatRequest::new(messages);

    // Python doesn't set temperature (uses LLM default)
    // Python uses structured outputs if supported, else JSON mode — we always use JSON mode
    let options = ChatOptions::default()
        .with_max_tokens(config.max_tokens as u32)
        .with_response_format(ChatResponseFormat::JsonMode);

    let genai_model = crate::llm::to_genai_model(&config.model);
    let mut last_error = String::new();
    for attempt in 0..config.max_retries {
        match genai_client
            .exec_chat(&genai_model, chat_req.clone(), Some(&options))
            .await
        {
            Ok(response) => {
                #[allow(deprecated)]
                let content = response
                    .content_text_as_str()
                    .unwrap_or("")
                    .to_string();
                return Ok(content);
            }
            Err(e) => {
                let err_str = e.to_string();
                let is_rate_limit = err_str.contains("429") || err_str.to_lowercase().contains("rate");
                let is_server_error = err_str.contains("500")
                    || err_str.contains("502")
                    || err_str.contains("503");

                if is_rate_limit {
                    let wait = 60u64 * 2u64.pow(attempt);
                    tracing::warn!("Rate limited, waiting {wait}s before retry (attempt {attempt})");
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    last_error = format!("Rate limited: {err_str}");
                } else if is_server_error {
                    let wait = 1u64 * 2u64.pow(attempt);
                    tracing::warn!("Server error, waiting {wait}s before retry (attempt {attempt})");
                    tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
                    last_error = format!("Server error: {err_str}");
                } else {
                    return Err(OneShotError::Llm(err_str));
                }
            }
        }
    }

    Err(OneShotError::Llm(format!(
        "All {0} retries exhausted. Last error: {last_error}",
        config.max_retries
    )))
}

fn parse_oneshot_response(text: &str) -> Result<Vec<OneShotAdSegment>, OneShotError> {
    // Try parsing as OneShotResponse directly
    if let Ok(resp) = serde_json::from_str::<OneShotResponse>(text) {
        return Ok(filter_valid_segments(resp.ad_segments));
    }

    // Try to find JSON in the response (LLM sometimes wraps in markdown)
    let json_str = extract_json(text)
        .ok_or_else(|| OneShotError::Parse("No JSON found in response".into()))?;

    // Try parsing the extracted JSON
    if let Ok(resp) = serde_json::from_str::<OneShotResponse>(json_str) {
        return Ok(filter_valid_segments(resp.ad_segments));
    }

    // Try repairing truncated JSON
    let repaired = repair_json(json_str);
    if let Ok(resp) = serde_json::from_str::<OneShotResponse>(&repaired) {
        return Ok(filter_valid_segments(resp.ad_segments));
    }

    // Last resort: try parsing as raw array
    if let Ok(segments) = serde_json::from_str::<Vec<OneShotAdSegment>>(json_str) {
        return Ok(filter_valid_segments(segments));
    }

    Err(OneShotError::Parse(format!("Could not parse response: {}", &text[..text.len().min(200)])))
}

fn filter_valid_segments(segments: Vec<OneShotAdSegment>) -> Vec<OneShotAdSegment> {
    segments
        .into_iter()
        .filter(|s| {
            if s.confidence > 1.0 || s.confidence < 0.0 {
                tracing::warn!("Skipping segment with invalid confidence: {}", s.confidence);
                return false;
            }
            if s.end_time <= s.start_time {
                tracing::warn!("Skipping segment with invalid times: {} - {}", s.start_time, s.end_time);
                return false;
            }
            true
        })
        .collect()
}

fn extract_json(text: &str) -> Option<&str> {
    // Find first { and last }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

fn repair_json(text: &str) -> String {
    let mut result = text.to_string();

    // Count unbalanced brackets
    let open_braces = result.chars().filter(|c| *c == '{').count();
    let close_braces = result.chars().filter(|c| *c == '}').count();
    let open_brackets = result.chars().filter(|c| *c == '[').count();
    let close_brackets = result.chars().filter(|c| *c == ']').count();

    // Add missing closing brackets/braces
    for _ in 0..(open_brackets.saturating_sub(close_brackets)) {
        result.push(']');
    }
    for _ in 0..(open_braces.saturating_sub(close_braces)) {
        result.push('}');
    }

    result
}

#[derive(Debug, thiserror::Error)]
pub enum OneShotError {
    #[error("LLM error: {0}")]
    Llm(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("database error: {0}")]
    Db(String),
}
