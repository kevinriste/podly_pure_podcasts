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
                tracing::warn!("Boundary refinement LLM call failed, trying heuristic: {e}");
                let (h_start, h_end) = heuristic_refine(block.start_time, block.end_time, &context);
                refined.push((h_start, h_end));
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
    use genai::chat::{ChatMessage, ChatOptions, ChatRequest};

    let genai_client = crate::llm::build_genai_client(
        &config.api_key,
        &config.model,
        config.base_url.as_deref(),
    )
    .map_err(|e| e.to_string())?;

    let chat_req = ChatRequest::new(vec![ChatMessage::user(prompt)]);

    let options = ChatOptions::default()
        .with_temperature(0.1)
        .with_max_tokens(4096u32);

    let genai_model = crate::llm::to_genai_model(&config.model);
    let response = genai_client
        .exec_chat(&genai_model, chat_req, Some(&options))
        .await
        .map_err(|e| format!("LLM error: {e}"))?;

    #[allow(deprecated)]
    let content = response
        .content_text_as_str()
        .unwrap_or("");

    // Parse JSON from response
    let start = content.find('{').ok_or("no JSON in refinement response")?;
    let json_str = &content[start..];
    let end = json_str.rfind('}').ok_or("no closing brace")? + 1;
    let json_str = &json_str[..end];

    serde_json::from_str::<RefinedBoundary>(json_str)
        .map_err(|e| format!("parse refinement response: {e}"))
}

/// Pattern-based heuristic refinement fallback (matches Python _heuristic_refine).
/// Extends ad boundaries into adjacent segments that contain intro/outro patterns.
fn heuristic_refine(ad_start: f64, ad_end: f64, context: &[Segment]) -> (f64, f64) {
    const INTRO_PATTERNS: &[&str] = &["brought to you", "sponsor", "let me tell you"];
    const OUTRO_PATTERNS: &[&str] = &[".com", "thanks to", "use code", "visit"];

    let mut refined_start = ad_start;
    let mut refined_end = ad_end;

    // Check segments before ad for intro patterns
    for seg in context {
        if seg.start_time < ad_start {
            let text_lower = seg.text.to_lowercase();
            if INTRO_PATTERNS.iter().any(|p| text_lower.contains(p)) {
                tracing::debug!("Heuristic: intro pattern matched at {:.1}s", seg.start_time);
                refined_start = seg.start_time;
            }
        }
    }

    // Check segments after ad for outro patterns
    for seg in context {
        if seg.start_time > ad_end {
            let text_lower = seg.text.to_lowercase();
            if OUTRO_PATTERNS.iter().any(|p| text_lower.contains(p)) {
                tracing::debug!("Heuristic: outro pattern matched at {:.1}s", seg.start_time);
                refined_end = seg.end_time;
            }
        }
    }

    // Constrain to max extension bounds
    refined_start = refined_start.max(ad_start - MAX_START_EXTENSION_SECONDS);
    refined_end = refined_end.min(ad_end + MAX_END_EXTENSION_SECONDS);

    if refined_start != ad_start || refined_end != ad_end {
        tracing::info!(
            "Heuristic refinement: {:.1}s-{:.1}s → {:.1}s-{:.1}s",
            ad_start, ad_end, refined_start, refined_end,
        );
    }

    (refined_start, refined_end)
}
