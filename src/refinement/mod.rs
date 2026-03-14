use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::classification::classifier::{ClassifierConfig, Segment};

// Boundary refinement prompt is built in code (not from template)
// since the original uses Jinja conditionals that can't be trivially substituted.

const MAX_START_EXTENSION_SECONDS: f64 = 30.0;
const MAX_END_EXTENSION_SECONDS: f64 = 15.0;
const CONTEXT_SEGMENTS: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedBoundary {
    pub refined_start: f64,
    pub refined_end: f64,
    pub start_adjustment_reason: Option<String>,
    pub end_adjustment_reason: Option<String>,
}

/// An ad block is a group of contiguous ad segments (within 10s gap).
#[derive(Debug)]
struct AdBlock {
    start_time: f64,
    end_time: f64,
    confidence: f64,
}

/// Refine ad boundaries using LLM-based analysis.
///
/// Groups detected ads into blocks, selects context segments around each block,
/// and calls the LLM to identify exact transition points.
pub async fn refine_boundaries(
    pool: &SqlitePool,
    post_id: i64,
    ad_segments: &[(f64, f64, f64)], // (start, end, confidence)
    transcript: &[Segment],
    config: &ClassifierConfig,
) -> Vec<(f64, f64)> {
    if ad_segments.is_empty() || transcript.is_empty() {
        return ad_segments.iter().map(|(s, e, _)| (*s, *e)).collect();
    }

    // Group into ad blocks (segments within 10s of each other)
    let blocks = group_ad_blocks(ad_segments);

    let mut refined = Vec::new();

    for block in &blocks {
        // Skip low-confidence or very short blocks
        if block.confidence < 0.6 || (block.end_time - block.start_time) < 15.0 {
            refined.push((block.start_time, block.end_time));
            continue;
        }

        // Get context segments around the block
        let context = get_context_segments(transcript, block.start_time, block.end_time);

        if context.is_empty() {
            refined.push((block.start_time, block.end_time));
            continue;
        }

        // Build refinement prompt
        let prompt = build_refinement_prompt(&context, block);

        // Call LLM
        match call_refinement_llm(config, &prompt).await {
            Ok(boundary) => {
                // Validate bounds
                let new_start = boundary.refined_start.max(block.start_time - MAX_START_EXTENSION_SECONDS);
                let new_end = boundary.refined_end.min(block.end_time + MAX_END_EXTENSION_SECONDS);

                if new_start < new_end {
                    tracing::info!(
                        "Refined ad block {:.1}s-{:.1}s → {:.1}s-{:.1}s (start: {:?}, end: {:?})",
                        block.start_time,
                        block.end_time,
                        new_start,
                        new_end,
                        boundary.start_adjustment_reason,
                        boundary.end_adjustment_reason,
                    );
                    refined.push((new_start, new_end));
                } else {
                    refined.push((block.start_time, block.end_time));
                }

                // Record model call
                let _ = sqlx::query(
                    "INSERT INTO model_call (post_id, first_segment_sequence_num, last_segment_sequence_num, model_name, prompt, response, timestamp, status)
                     VALUES (?, 0, 0, ?, ?, ?, ?, 'success')"
                )
                .bind(post_id)
                .bind(format!("{}_refinement", config.model))
                .bind(&prompt)
                .bind(serde_json::to_string(&boundary).unwrap_or_default())
                .bind(chrono::Utc::now().to_rfc3339())
                .execute(pool)
                .await;
            }
            Err(e) => {
                tracing::warn!("Boundary refinement LLM call failed: {e}");
                refined.push((block.start_time, block.end_time));
            }
        }
    }

    refined
}

fn group_ad_blocks(ads: &[(f64, f64, f64)]) -> Vec<AdBlock> {
    let mut blocks: Vec<AdBlock> = Vec::new();

    for &(start, end, confidence) in ads {
        if let Some(last) = blocks.last_mut() {
            if start - last.end_time <= 10.0 {
                // Merge into existing block
                last.end_time = end;
                last.confidence = last.confidence.max(confidence);
                continue;
            }
        }
        blocks.push(AdBlock {
            start_time: start,
            end_time: end,
            confidence,
        });
    }

    blocks
}

fn get_context_segments(transcript: &[Segment], ad_start: f64, ad_end: f64) -> Vec<Segment> {
    // Find segments around the ad block (±CONTEXT_SEGMENTS segments)
    let start_idx = transcript
        .iter()
        .position(|s| s.start_time >= ad_start)
        .unwrap_or(0);
    let end_idx = transcript
        .iter()
        .position(|s| s.start_time > ad_end)
        .unwrap_or(transcript.len());

    let context_start = start_idx.saturating_sub(CONTEXT_SEGMENTS);
    let context_end = (end_idx + CONTEXT_SEGMENTS).min(transcript.len());

    transcript[context_start..context_end].to_vec()
}

fn build_refinement_prompt(context: &[Segment], block: &AdBlock) -> String {
    let mut segments_text = String::new();
    for seg in context {
        segments_text.push_str(&format!("[{}] {}\n", seg.start_time, seg.text));
    }

    let confidence_guidance = if block.confidence > 0.9 {
        "be aggressive with boundary extension"
    } else if block.confidence > 0.7 {
        "be conservative, only extend for clear signals"
    } else {
        "minimal changes, preserve content"
    };

    format!(
        r#"You are analyzing podcast transcript segments to precisely identify advertisement boundaries.

**Detected Ad Block**: {start:.1}s - {end:.1}s
**Original Confidence**: {confidence:.2}

**CONTEXT SEGMENTS**:
{segments}
**BOUNDARY DETECTION RULES**:
- AD START: Sponsor introductions, transition phrases, host acknowledgments
- AD END: Sponsor conclusions, final CTAs, transitions back to content
- CONTENT RESUMPTION: Natural conversation, topic changes, interview continuation

**For confidence {confidence:.2}**: {guidance}

**OUTPUT FORMAT**:
Respond with valid JSON:
{{"refined_start": {start:.1}, "refined_end": {end:.1}, "start_adjustment_reason": "reason", "end_adjustment_reason": "reason"}}

If no refinement needed, return original timestamps with "No adjustment needed" reasons.
Ensure refined_start < refined_end. Keep adjustments close to detected timestamps."#,
        start = block.start_time,
        end = block.end_time,
        confidence = block.confidence,
        segments = segments_text,
        guidance = confidence_guidance,
    )
}

async fn call_refinement_llm(
    config: &ClassifierConfig,
    prompt: &str,
) -> Result<RefinedBoundary, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(config.timeout_sec))
        .build()
        .map_err(|e| e.to_string())?;

    let base_url = config
        .base_url
        .as_deref()
        .unwrap_or("https://api.openai.com/v1");

    let body = serde_json::json!({
        "model": config.model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 500,
    });

    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err(format!("LLM error: {}", resp.status()));
    }

    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");

    // Parse JSON from response
    let start = content.find('{').ok_or("no JSON in refinement response")?;
    let json_str = &content[start..];
    let end = json_str.rfind('}').ok_or("no closing brace")? + 1;
    let json_str = &json_str[..end];

    serde_json::from_str::<RefinedBoundary>(json_str)
        .map_err(|e| format!("parse refinement response: {e}"))
}
