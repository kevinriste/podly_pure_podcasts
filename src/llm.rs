use genai::chat::{ChatResponseFormat, JsonSpec};
use genai::resolver::AuthData;
use genai::{Client, ServiceTarget};
use serde_json::json;

/// Convert litellm-style model names (e.g. "gemini/gemini-2.0-flash") to
/// genai-style namespace (e.g. "gemini::gemini-2.0-flash").
///
/// genai uses "::" as the namespace separator, while litellm/Python uses "/".
/// Only converts the FIRST "/" — model names like "meta-llama/Llama-3" keep
/// their internal slashes.
pub fn to_genai_model(model: &str) -> String {
    if model.contains("::") {
        // Already in genai format
        return model.to_string();
    }
    if let Some(slash_pos) = model.find('/') {
        let prefix = &model[..slash_pos];
        let rest = &model[slash_pos + 1..];
        format!("{prefix}::{rest}")
    } else {
        model.to_string()
    }
}

/// Build a genai Client configured with the given API key and optional base URL.
///
/// The model name can use litellm-style prefixes (e.g., "gemini/gemini-2.0-flash",
/// "openai/gpt-4o") — they are automatically converted to genai's "::" format.
pub fn build_genai_client(
    api_key: &str,
    _model: &str,
    base_url: Option<&str>,
) -> Result<Client, genai::Error> {
    let api_key = api_key.to_string();
    let base_url = base_url.map(|s| s.to_string());

    let mut builder = Client::builder();

    // Always provide the API key via AuthResolver (overrides env vars)
    builder = builder.with_auth_resolver_fn(move |_model_iden| {
        Ok(Some(AuthData::Key(api_key.clone())))
    });

    // If a custom base URL is provided, override the service target endpoint
    if let Some(url) = base_url {
        if !url.is_empty() {
            builder = builder.with_service_target_resolver_fn(move |mut st: ServiceTarget| {
                st.endpoint = genai::resolver::Endpoint::from_owned(url.clone());
                Ok(st)
            });
        }
    }

    Ok(builder.build())
}

/// Check if a model supports structured outputs (JSON schema response format).
///
/// Mirrors Python's `model_supports_structured_outputs()` which delegates to
/// litellm's `supports_response_schema()`. Since we don't have litellm's model
/// database, we check known provider/model patterns.
pub fn model_supports_structured_outputs(model: &str) -> bool {
    let lower = model.to_lowercase();

    // Gemini models support responseJsonSchema
    if lower.starts_with("gemini/") || lower.starts_with("gemini::") {
        return true;
    }

    // OpenAI gpt-4o, gpt-5, o1, o3 families support json_schema
    if lower.starts_with("openai/") || lower.starts_with("openai::") {
        let model_part = lower.split(['/', ':']).last().unwrap_or("");
        return model_part.starts_with("gpt-4o")
            || model_part.starts_with("gpt-5")
            || model_part.starts_with("o1")
            || model_part.starts_with("o3");
    }

    // Models without a provider prefix — check the name directly
    if !lower.contains('/') && !lower.contains("::") {
        let starts_with_known = lower.starts_with("gpt-4o")
            || lower.starts_with("gpt-5")
            || lower.starts_with("o1")
            || lower.starts_with("o3")
            || lower.starts_with("gemini");
        return starts_with_known;
    }

    false
}

/// Build the response format for oneshot classification.
///
/// Uses structured outputs (JsonSpec) when the model supports it,
/// otherwise falls back to JsonMode.
pub fn oneshot_response_format(model: &str) -> ChatResponseFormat {
    if model_supports_structured_outputs(model) {
        tracing::info!("Using structured outputs for model {model}");
        ChatResponseFormat::JsonSpec(JsonSpec::new(
            "oneshot_response",
            json!({
                "type": "object",
                "properties": {
                    "ad_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_time": {
                                    "type": "number",
                                    "description": "Exact start time in seconds"
                                },
                                "end_time": {
                                    "type": "number",
                                    "description": "Exact end time in seconds"
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score from 0.0 to 1.0"
                                },
                                "ad_type": {
                                    "type": ["string", "null"],
                                    "description": "Type of ad: sponsor, house_ad, or transition"
                                },
                                "reason": {
                                    "type": ["string", "null"],
                                    "description": "Brief explanation for why this segment is classified as an ad"
                                }
                            },
                            "required": ["start_time", "end_time", "confidence", "ad_type", "reason"]
                        }
                    }
                },
                "required": ["ad_segments"]
            }),
        ))
    } else {
        tracing::info!("Using JSON mode fallback for model {model}");
        ChatResponseFormat::JsonMode
    }
}

/// Check if a model requires `max_completion_tokens` instead of `max_tokens`.
/// Matches Python's `model_uses_max_completion_tokens()` from shared/llm_utils.py.
/// OpenAI deprecated `max_tokens` for newer models (gpt-4o, gpt-5, o1, etc.)
pub fn model_uses_max_completion_tokens(model: &str) -> bool {
    let lower = model.to_lowercase();
    const PATTERNS: &[&str] = &["gpt-5", "gpt-4o", "o1-", "o1_", "o1/", "chatgpt-4o-latest"];
    PATTERNS.iter().any(|p| lower.contains(p))
}

/// Count tokens in a text string using tiktoken.
/// Falls back to chars/4 estimate if the model isn't recognized.
pub fn count_tokens(text: &str, model: &str) -> usize {
    // Try to get a BPE tokenizer for the model
    match tiktoken_rs::get_bpe_from_model(model) {
        Ok(bpe) => bpe.encode_with_special_tokens(text).len(),
        Err(_) => {
            // Try cl100k_base (GPT-4 default) as fallback
            match tiktoken_rs::cl100k_base() {
                Ok(bpe) => bpe.encode_with_special_tokens(text).len(),
                Err(_) => text.len() / 4, // rough estimate
            }
        }
    }
}

/// Execute a chat completion and return the response text.
#[allow(dead_code)]
pub async fn chat_completion(
    client: &Client,
    model: &str,
    system_prompt: Option<&str>,
    user_prompt: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
) -> Result<String, genai::Error> {
    use genai::chat::{ChatMessage, ChatOptions, ChatRequest};

    let genai_model = to_genai_model(model);

    let mut messages = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(ChatMessage::system(sys));
    }
    messages.push(ChatMessage::user(user_prompt));

    let chat_req = ChatRequest::new(messages);

    let mut options = ChatOptions::default();
    if let Some(t) = temperature {
        options.temperature = Some(t);
    }
    if let Some(mt) = max_tokens {
        options.max_tokens = Some(mt);
    }

    let response = client.exec_chat(&genai_model, chat_req, Some(&options)).await?;
    #[allow(deprecated)]
    let text = response
        .content_text_as_str()
        .unwrap_or("")
        .to_string();
    Ok(text)
}
