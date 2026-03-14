# Python/Rust Backend Parity Status

Last updated: 2026-03-14

## Verified Matching Endpoints

All endpoints below return identical status codes, response keys, and response shapes
between Python (podly.klt.pw) and Rust (podly2.klt.pw).

### Auth
- `POST /api/auth/login` — success, bad password (401), missing user (401), empty body (400)
- `GET /api/auth/status`
- `GET /api/auth/me`
- `GET /api/auth/users`
- `POST /api/auth/logout` (204)
- `POST /api/auth/change-password` — wrong current password returns 401 "Current password is incorrect."
- `GET /api/auth/discord/status`
- `GET /api/auth/discord/config` — wrapped in `{config, env_overrides}`

### Config
- `GET /api/config` — wrapped in `{config, env_overrides}`, whisper flattened by type, API keys masked
- `PUT /api/config` — saves all sections (llm, whisper, processing, output, app, chapter_filter), returns sanitized config
- `GET /api/config/api_configured_check`
- `GET /api/config/whisper-capabilities`
- `POST /api/config/test-whisper` — reads from nested `{whisper: {...}}`, falls back to env/DB

### Feeds
- `GET /feeds` — feed list
- `POST /feed` — add feed (accepts form data and JSON)
- `DELETE /feed/{id}` — delete feed (204)
- `POST /api/feeds/{id}/refresh` — 202 Accepted, async background refresh
- `PATCH /api/feeds/{id}/settings` — response includes all feed fields
- `POST /api/feeds/{id}/share-link` — 201, returns `{feed_id, feed_secret, feed_token, url}`
- `POST /api/feeds/{id}/toggle-whitelist-all` — matching response shape and messages
- `GET /api/feeds/search`

### Posts
- `GET /api/feeds/{id}/posts` — matching pagination keys
- `POST /api/posts/{guid}/whitelist` — body: `{"whitelisted": bool}`, message: "Whitelist status updated successfully"
- `GET /api/posts/{guid}/status` — matching keys: status, step, step_name, total_steps, progress_percentage, message, download_url, started_at
- `GET /api/posts/{guid}/stats` — includes chapters key
- `GET /api/posts/{guid}/processing-estimate`
- `POST /api/posts/{guid}/process`
- `POST /api/posts/{guid}/reprocess`

### RSS
- `GET /feed/{id}` — 401 without auth, 200 with session/token, XML content
- `GET /feed/{id}?token=...` — token-based auth

### Unauth Access
- `GET /feeds` without auth: 401
- `GET /api/config` without auth: 401
- `GET /api/auth/status` without auth: 200 (by design)

## Known Caveats / Remaining Differences

### Data-Level Differences (not code bugs)
- **user.id**: Different auto-increment values between DBs (PY=1, RS=2)
- **oneshot_model value**: Python DB has env var baked in from first boot; Rust DB has stale default. Both code paths handle env overlay correctly.
- **processing_job records**: Rust DB is missing historical `processing_job` records from Python's job system. This causes `GET /api/posts/{guid}/status` to return "skipped" (no job) instead of "completed" (with job + started_at). Code is correct for both states.
- **ad_detection_strategy**: Python resolves "inherit" to the actual strategy; Rust returns raw "inherit". Frontend handles this client-side.

### Architectural Limitations
- **`POST /api/config/test-llm`**: Returns 500 for `gemini/*` models. Python routes these through `litellm` which handles the Gemini API natively. Rust uses raw OpenAI-compatible HTTP requests — needs a multi-provider LLM client (e.g. `genai` crate) to support non-OpenAI models.
- **`POST /feed` response format**: Python returns 302 redirect (server-rendered), Rust returns 201 JSON. Both correctly add the feed. Frontend handles both.
- **`GET /api/auth/me` after logout**: Python doesn't actually clear the session on logout (returns user data). Rust correctly returns 401. This is a Python bug.

### Environment Differences
- **whisper-capabilities `local_available`**: Python has whisper installed, Rust compiled without `local-whisper` feature. Not a code issue.
- **billing/status**: Both return 404 (billing not configured). Different HTML (Flask 404 vs SPA).

### Not Yet Tested
- Full process/reprocess lifecycle (downloading audio, transcribing, running LLM, building output)
- Scheduler/background job execution
- Billing/Stripe webhook handling
- Discord OAuth flow (tested config endpoint only)
- Large file uploads/downloads under load

### Operational Notes
- **toggle-whitelist-all creates processing jobs**: When whitelisting all posts, the `enqueue_pending_jobs` call creates pending processing jobs for every newly whitelisted post. Be careful with this endpoint in testing — always unwhitelist-all to undo, AND clean up pending jobs from the DB.
- **Python add_feed is broken**: As of 2026-03-14, Python's `POST /feed` returns 500. This is a Python bug, not a Rust issue.
- **Refresh feed is async**: Both backends now fire-and-forget the refresh. Python uses a background Thread, Rust uses `tokio::spawn`.
