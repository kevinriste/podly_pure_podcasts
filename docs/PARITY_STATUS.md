# Python/Rust Backend Parity Status

Last updated: 2026-03-15

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

### Architectural Choices (By Design)
- **LLM provider**: `genai` crate (Rust) vs `litellm` (Python). Both support OpenAI, Anthropic, Gemini, Groq. See DECISION-020.
- **`POST /feed` response format**: Python returns 302 redirect (server-rendered), Rust returns 201 JSON. Both correctly add the feed. Frontend handles both.
- **`GET /api/auth/me` after logout**: Python doesn't actually clear the session on logout (returns user data). Rust correctly returns 401. This is a Python bug.
- **`search_feeds`**: PodcastIndex API (Rust) vs iTunes (Python) — same result format.
- **Ad merging**: Proximity-only (Rust) vs proximity + content-aware keyword extraction (Python). See DECISION-034.
- **Session store**: Custom SQLite-backed (Rust) vs Flask server-side (Python). Both persist across restarts. See DECISION-032.
- **Password hashing**: Argon2id (Rust) with bcrypt fallback for legacy hashes. See DECISION-015.

### Pipeline Differences (Accepted)
- Content-aware ad merging (keyword/sponsor/URL detection) — not implemented (DECISION-034)
- No proactive token rate limiting (reactive 429 backoff only)
- `max_tokens` instead of `max_completion_tokens` for newer OpenAI models (DECISION-036)
- No pre-reprocess snapshot creation
- No input token count validation/trimming for oversized prompts

### Environment Differences
- **whisper-capabilities `local_available`**: Python has whisper installed, Rust compiled without `local-whisper` feature. Not a code issue.

### Not Yet Tested
- Full process/reprocess lifecycle (downloading audio, transcribing, running LLM, building output)
- Scheduler/background job execution
- Billing/Stripe webhook handling
- Discord OAuth flow (tested config endpoint only)
- Large file uploads/downloads under load
