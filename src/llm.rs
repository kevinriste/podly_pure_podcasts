use genai::resolver::AuthData;
use genai::{Client, ServiceTarget};

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
