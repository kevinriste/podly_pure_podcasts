use genai::resolver::AuthData;
use genai::{Client, ServiceTarget};

/// Build a genai Client configured with the given API key and optional base URL.
///
/// The model name can use litellm-style prefixes (e.g., "gemini/gemini-3.1-flash",
/// "openai/gpt-4o") and genai will route to the correct provider automatically.
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

    let response = client.exec_chat(model, chat_req, Some(&options)).await?;
    #[allow(deprecated)]
    let text = response
        .content_text_as_str()
        .unwrap_or("")
        .to_string();
    Ok(text)
}
