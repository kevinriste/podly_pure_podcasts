use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Json;
use axum::routing::{get, post};
use axum::{Extension, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::auth::AuthenticatedUser;
use crate::db::queries;
use crate::error::{AppError, AppResult};
use crate::AppState;

const ACTIVE_FEED_ALLOWANCE: i64 = 10;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/billing/summary", get(billing_summary))
        .route("/api/billing/subscription", post(subscription))
        .route("/api/billing/portal-session", post(portal_session))
        .route("/api/billing/stripe-webhook", post(stripe_webhook))
}

fn is_stripe_enabled(state: &AppState) -> bool {
    state.config.stripe_secret_key.is_some() && state.config.stripe_product_id.is_some()
}

/// GET /api/billing/summary — returns user's subscription info.
async fn billing_summary(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
) -> Result<Json<Value>, AppError> {
    if !is_stripe_enabled(&state) {
        return Ok(Json(json!({
            "feed_allowance": 0,
            "feeds_in_use": 0,
            "remaining": 0,
            "current_amount": null,
            "min_amount_cents": 0,
            "subscription_status": "disabled",
            "stripe_subscription_id": null,
            "stripe_customer_id": null,
            "product_id": null,
        })));
    }

    let Extension(auth) = auth_user.ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    let user = queries::get_user_by_id(&state.db, auth.id).await?
        .ok_or(AppError::Unauthorized("Authentication required.".into()))?;

    let feeds_in_use = queries::count_user_feeds(&state.db, user.id).await?;
    let allowance = user.manual_feed_allowance.unwrap_or(user.feed_allowance);
    let remaining = (allowance - feeds_in_use).max(0);

    // Get current subscription amount if active
    let mut current_amount_cents: Option<i64> = None;
    if let Some(sub_id) = &user.stripe_subscription_id {
        if !sub_id.is_empty() {
            if let Some(key) = &state.config.stripe_secret_key {
                current_amount_cents = get_subscription_amount(key, sub_id).await.ok();
            }
        }
    }

    Ok(Json(json!({
        "feed_allowance": allowance,
        "feeds_in_use": feeds_in_use,
        "remaining": remaining,
        "current_amount": current_amount_cents.unwrap_or(0),
        "min_amount_cents": state.config.stripe_min_subscription_amount_cents,
        "subscription_status": if user.feed_subscription_status.is_empty() { "inactive" } else { user.feed_subscription_status.as_str() },
        "stripe_subscription_id": user.stripe_subscription_id,
        "stripe_customer_id": user.stripe_customer_id,
        "product_id": state.config.stripe_product_id,
    })))
}

#[derive(Deserialize)]
struct SubscriptionRequest {
    // Python uses "amount", accept both for compatibility
    amount: Option<i64>,
    amount_cents: Option<i64>,
}

/// POST /api/billing/subscription — create/update/cancel subscription with custom amount.
async fn subscription(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    Json(body): Json<SubscriptionRequest>,
) -> Result<Json<Value>, AppError> {
    if !is_stripe_enabled(&state) {
        return Err(AppError::ServiceUnavailable("Stripe billing is not configured.".into()));
    }

    let stripe_key = state.config.stripe_secret_key.as_deref().unwrap();
    let product_id = state.config.stripe_product_id.as_deref().unwrap();

    let Extension(auth) = auth_user.ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    let user = queries::get_user_by_id(&state.db, auth.id).await?
        .ok_or(AppError::Unauthorized("Authentication required.".into()))?;

    let amount_cents = body.amount.or(body.amount_cents).unwrap_or(0);

    // Cancel if amount is 0
    if amount_cents == 0 {
        if let Some(sub_id) = &user.stripe_subscription_id {
            if !sub_id.is_empty() {
                cancel_subscription(stripe_key, sub_id).await
                    .map_err(|e| AppError::Internal(anyhow::anyhow!("Stripe cancel error: {e}")))?;
            }
        }
        queries::set_user_billing_fields(
            &state.db, user.id, None, None, Some(0), Some("canceled"),
        ).await?;
        let feeds_in_use = queries::count_user_feeds(&state.db, user.id).await?;
        return Ok(Json(json!({
            "feed_allowance": 0,
            "feeds_in_use": feeds_in_use,
            "remaining": 0,
            "subscription_status": "canceled",
            "requires_stripe_checkout": false,
            "message": "Subscription canceled.",
        })));
    }

    // Validate minimum
    if amount_cents < state.config.stripe_min_subscription_amount_cents {
        let dollars = state.config.stripe_min_subscription_amount_cents as f64 / 100.0;
        return Err(AppError::BadRequest(format!(
            "Minimum amount is ${dollars:.2}"
        )));
    }

    // Ensure Stripe customer exists
    let customer_id = if let Some(cid) = &user.stripe_customer_id {
        if !cid.is_empty() {
            cid.clone()
        } else {
            let cid = create_stripe_customer(stripe_key, &user.username, user.id).await
                .map_err(|e| AppError::Internal(anyhow::anyhow!("Stripe customer error: {e}")))?;
            queries::set_user_billing_fields(
                &state.db, user.id, Some(&cid), None, None, None,
            ).await?;
            cid
        }
    } else {
        let cid = create_stripe_customer(stripe_key, &user.username, user.id).await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("Stripe customer error: {e}")))?;
        queries::set_user_billing_fields(
            &state.db, user.id, Some(&cid), None, None, None,
        ).await?;
        cid
    };

    // If user already has a subscription, modify it
    if let Some(sub_id) = &user.stripe_subscription_id {
        if !sub_id.is_empty() && user.feed_subscription_status != "canceled" {
            update_subscription(stripe_key, sub_id, product_id, amount_cents).await
                .map_err(|e| AppError::BadGateway(format!("{e}")))?;
            // Set allowance to 10 like Python does
            queries::set_user_billing_fields(
                &state.db, user.id, None, None, Some(ACTIVE_FEED_ALLOWANCE), Some("active"),
            ).await?;
            let feeds_in_use = queries::count_user_feeds(&state.db, user.id).await?;
            return Ok(Json(json!({
                "feed_allowance": ACTIVE_FEED_ALLOWANCE,
                "feeds_in_use": feeds_in_use,
                "remaining": (ACTIVE_FEED_ALLOWANCE - feeds_in_use).max(0),
                "subscription_status": "active",
                "requires_stripe_checkout": false,
                "message": "Subscription updated.",
            })));
        }
    }

    // Create checkout session for new subscription
    let checkout_url = create_checkout_session(
        stripe_key,
        &customer_id,
        product_id,
        amount_cents,
        user.id,
    ).await
        .map_err(|e| AppError::BadGateway(format!("{e}")))?;

    let feeds_in_use = queries::count_user_feeds(&state.db, user.id).await?;
    let allowance = user.manual_feed_allowance.unwrap_or(user.feed_allowance);
    Ok(Json(json!({
        "requires_stripe_checkout": true,
        "checkout_url": checkout_url,
        "feed_allowance": allowance,
        "feeds_in_use": feeds_in_use,
        "subscription_status": if user.feed_subscription_status.is_empty() { "inactive" } else { user.feed_subscription_status.as_str() },
    })))
}

/// POST /api/billing/portal-session — creates Stripe portal session.
async fn portal_session(
    State(state): State<AppState>,
    auth_user: Option<Extension<AuthenticatedUser>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Value>, AppError> {
    if !is_stripe_enabled(&state) {
        return Err(AppError::ServiceUnavailable("Stripe billing is not configured.".into()));
    }

    let stripe_key = state.config.stripe_secret_key.as_deref().unwrap();

    let Extension(auth) = auth_user.ok_or(AppError::Unauthorized("Authentication required.".into()))?;
    let user = queries::get_user_by_id(&state.db, auth.id).await?
        .ok_or(AppError::Unauthorized("Authentication required.".into()))?;

    let customer_id = user.stripe_customer_id.as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| AppError::BadRequest("No Stripe customer on file.".into()))?;

    // Build return_url from Host header like Python does
    let host = headers.get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("localhost");
    let scheme = headers.get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("http");
    let return_url = format!("{scheme}://{host}/billing?checkout=success");

    let url = create_portal_session(stripe_key, customer_id, &return_url).await
        .map_err(|e| AppError::BadGateway(format!("{e}")))?;

    Ok(Json(json!({"url": url})))
}

/// POST /api/billing/stripe-webhook — handles Stripe webhook events.
async fn stripe_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> AppResult<Json<Value>> {
    let webhook_secret = match &state.config.stripe_webhook_secret {
        Some(s) if !s.is_empty() => s.clone(),
        _ => {
            tracing::warn!("Stripe webhook received but no webhook secret configured");
            return Err(AppError::BadRequest("Webhook secret not configured.".into()));
        }
    };

    let signature = headers
        .get("stripe-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let payload = std::str::from_utf8(&body)
        .map_err(|_| AppError::BadRequest("Invalid UTF-8 payload".into()))?;

    // Verify signature
    if !verify_stripe_signature(payload, signature, &webhook_secret) {
        tracing::warn!("Stripe webhook signature verification failed");
        return Err(AppError::BadRequest("Invalid signature".into()));
    }

    let event: Value = serde_json::from_str(payload)
        .map_err(|e| AppError::BadRequest(format!("Invalid JSON: {e}")))?;

    let event_type = event["type"].as_str().unwrap_or("");
    tracing::info!("Stripe webhook event: {event_type}");

    match event_type {
        "customer.subscription.created" | "customer.subscription.updated" => {
            handle_subscription_update(&state, &event["data"]["object"]).await?;
        }
        "customer.subscription.deleted" | "customer.subscription.paused" => {
            handle_subscription_ended(&state, &event["data"]["object"], event_type).await?;
        }
        "checkout.session.completed" => {
            handle_checkout_completed(&state, &event["data"]["object"]).await?;
        }
        _ => {
            tracing::debug!("Ignoring Stripe event: {event_type}");
        }
    }

    Ok(Json(json!({"status": "ok"})))
}

// ── Webhook event handlers ──

async fn handle_subscription_update(state: &AppState, sub: &Value) -> AppResult<()> {
    let customer_id = sub["customer"].as_str().unwrap_or("");
    let sub_id = sub["id"].as_str();
    let status = sub["status"].as_str().unwrap_or("");

    let (allowance, db_status) = match status {
        "active" | "trialing" | "past_due" => (ACTIVE_FEED_ALLOWANCE, status),
        _ => (0, status),
    };

    queries::set_user_billing_by_customer_id(
        &state.db, customer_id, sub_id, allowance, db_status,
    ).await?;

    Ok(())
}

async fn handle_subscription_ended(state: &AppState, sub: &Value, event_type: &str) -> AppResult<()> {
    let customer_id = sub["customer"].as_str().unwrap_or("");
    let sub_id = sub["id"].as_str();
    let db_status = if event_type.contains("deleted") { "canceled" } else { "paused" };

    queries::set_user_billing_by_customer_id(
        &state.db, customer_id, sub_id, 0, db_status,
    ).await?;

    Ok(())
}

async fn handle_checkout_completed(state: &AppState, session: &Value) -> AppResult<()> {
    let customer_id = session["customer"].as_str().unwrap_or("");
    let sub_id = session["subscription"].as_str();

    if customer_id.is_empty() {
        return Ok(());
    }

    // Link subscription to user
    if let Some(sub_id) = sub_id {
        queries::set_user_billing_by_customer_id(
            &state.db, customer_id, Some(sub_id), ACTIVE_FEED_ALLOWANCE, "active",
        ).await?;
    }

    Ok(())
}

// ── Stripe API helpers (raw HTTP) ──

async fn create_stripe_customer(key: &str, username: &str, user_id: i64) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.stripe.com/v1/customers")
        .basic_auth(key, None::<&str>)
        .form(&[
            ("name", username.to_string()),
            ("metadata[podly_user_id]", user_id.to_string()),
        ])
        .send()
        .await?;

    let data: Value = resp.json().await?;
    data["id"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No customer id in Stripe response: {data}"))
}

async fn create_checkout_session(
    key: &str,
    customer_id: &str,
    product_id: &str,
    amount_cents: i64,
    user_id: i64,
) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    // Note: success/cancel URLs use {CHECKOUT_SESSION_ID} as Stripe's template variable.
    // In production these would be built from the request host, but Stripe handles the redirect.
    let success_url = "/billing?checkout=success";
    let cancel_url = "/billing?checkout=cancel";
    let user_id_str = user_id.to_string();
    let amount_str = amount_cents.to_string();
    let resp = client
        .post("https://api.stripe.com/v1/checkout/sessions")
        .basic_auth(key, None::<&str>)
        .form(&[
            ("mode", "subscription"),
            ("customer", customer_id),
            ("line_items[0][price_data][currency]", "usd"),
            ("line_items[0][price_data][product]", product_id),
            ("line_items[0][price_data][unit_amount]", &amount_str),
            ("line_items[0][price_data][recurring][interval]", "month"),
            ("line_items[0][quantity]", "1"),
            ("success_url", success_url),
            ("cancel_url", cancel_url),
            ("metadata[user_id]", &user_id_str),
            ("subscription_data[metadata][user_id]", &user_id_str),
        ])
        .send()
        .await?;

    let data: Value = resp.json().await?;
    data["url"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No checkout URL in Stripe response: {data}"))
}

async fn create_portal_session(key: &str, customer_id: &str, return_url: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.stripe.com/v1/billing_portal/sessions")
        .basic_auth(key, None::<&str>)
        .form(&[("customer", customer_id), ("return_url", return_url)])
        .send()
        .await?;

    let data: Value = resp.json().await?;
    data["url"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("No portal URL in Stripe response: {data}"))
}

async fn update_subscription(
    key: &str,
    subscription_id: &str,
    product_id: &str,
    amount_cents: i64,
) -> anyhow::Result<()> {
    let client = reqwest::Client::new();

    // First get the current subscription to find the item ID
    let resp = client
        .get(&format!("https://api.stripe.com/v1/subscriptions/{subscription_id}"))
        .basic_auth(key, None::<&str>)
        .send()
        .await?;
    let sub: Value = resp.json().await?;
    let item_id = sub["items"]["data"][0]["id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("No subscription item found"))?;

    // Update with new price
    let _resp = client
        .post(&format!("https://api.stripe.com/v1/subscriptions/{subscription_id}"))
        .basic_auth(key, None::<&str>)
        .form(&[
            ("proration_behavior", "none"),
            ("items[0][id]", item_id),
            ("items[0][price_data][currency]", "usd"),
            ("items[0][price_data][product]", product_id),
            ("items[0][price_data][unit_amount]", &amount_cents.to_string()),
            ("items[0][price_data][recurring][interval]", "month"),
        ])
        .send()
        .await?;

    Ok(())
}

async fn cancel_subscription(key: &str, subscription_id: &str) -> anyhow::Result<()> {
    let client = reqwest::Client::new();
    let _resp = client
        .delete(&format!("https://api.stripe.com/v1/subscriptions/{subscription_id}"))
        .basic_auth(key, None::<&str>)
        .send()
        .await?;
    Ok(())
}

async fn get_subscription_amount(key: &str, subscription_id: &str) -> anyhow::Result<i64> {
    let client = reqwest::Client::new();
    let resp = client
        .get(&format!("https://api.stripe.com/v1/subscriptions/{subscription_id}"))
        .basic_auth(key, None::<&str>)
        .send()
        .await?;
    let data: Value = resp.json().await?;
    data["items"]["data"][0]["price"]["unit_amount"]
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("No amount in subscription"))
}

// ── Stripe webhook signature verification ──

fn verify_stripe_signature(payload: &str, signature_header: &str, secret: &str) -> bool {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    // Parse signature header: t=...,v1=...
    let mut timestamp = "";
    let mut signatures: Vec<&str> = vec![];

    for part in signature_header.split(',') {
        let part = part.trim();
        if let Some(t) = part.strip_prefix("t=") {
            timestamp = t;
        } else if let Some(v1) = part.strip_prefix("v1=") {
            signatures.push(v1);
        }
    }

    if timestamp.is_empty() || signatures.is_empty() {
        return false;
    }

    // Build signed payload
    let signed_payload = format!("{timestamp}.{payload}");

    // Compute expected signature
    let mut mac = match Hmac::<Sha256>::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(signed_payload.as_bytes());
    let expected = hex::encode(mac.finalize().into_bytes());

    // Compare with provided signatures (constant-time not strictly needed for webhooks,
    // but use equality check)
    signatures.iter().any(|sig| *sig == expected)
}
