# Podly Rust Backend — Decision Log

## DECISION-001: Project layout
**Date:** 2026-03-13
**Category:** architecture
**Context:** The repo already has a `src/` directory with Python code. We need to add Rust code.
**Decision:** Rust code coexists alongside Python code in the same repo. `Cargo.toml` at root, Rust source in `src/main.rs` etc. Cargo only compiles `.rs` files so Python subdirectories (`src/app/`, `src/podcast_processor/`, `src/shared/`) are ignored by the Rust toolchain.
**Rationale:** Simplest approach. Once the Rust rewrite is complete, Python code can be removed. No need for a subdirectory or workspace restructuring.

## DECISION-002: Timestamps as TEXT (ISO 8601) in SQLite
**Date:** 2026-03-13
**Category:** schema
**Context:** SQLite has no native datetime type. Python used `db.DateTime` which SQLAlchemy stores as TEXT.
**Python behavior:** Stores datetimes as ISO 8601 strings via SQLAlchemy.
**Rust behavior:** Stores as TEXT in ISO 8601 format, parsed via `chrono::NaiveDateTime` or `chrono::DateTime<Utc>`.
**Rationale:** Matches Python behavior, human-readable, sortable, and compatible with legacy data migration.

## DECISION-003: Settings tables kept as singletons
**Date:** 2026-03-13
**Category:** schema
**Context:** Python has singleton settings tables (LLMSettings, WhisperSettings, etc.) with `id=1`.
**Python behavior:** Single row per settings table, accessed by `id=1`.
**Rust behavior:** Same pattern. Query by `id=1`, upsert on save.
**Rationale:** Direct compatibility with existing data and migration tool.

## DECISION-004: UserFeed table name kept as "feed_supporter"
**Date:** 2026-03-13
**Category:** schema
**Context:** Python model `UserFeed` uses `__tablename__ = "feed_supporter"`.
**Python behavior:** Table is named `feed_supporter`.
**Rust behavior:** Same table name `feed_supporter` to enable legacy data migration.
**Rationale:** Migration compatibility. The name is misleading but changing it would break the migration tool unnecessarily.

## DECISION-005: RSS parsing with feed-rs
**Date:** 2026-03-13
**Category:** crate
**Context:** Need to parse RSS/Atom feeds. Options: `rss` crate (RSS-only), `feed-rs` (RSS + Atom + JSON Feed).
**Decision:** Use `feed-rs` for parsing incoming podcast feeds.
**Rationale:** `feed-rs` handles RSS 2.0, Atom, and JSON Feed formats. Podcasts may use any of these. `rss` crate only handles RSS 2.0. `quick-xml` used separately for RSS generation (ad-free feed output) since we need precise control over the XML structure.

## DECISION-006: Preserving Python column name typo
**Date:** 2026-03-13
**Category:** schema
**Context:** Python has `min_ad_segement_separation_seconds` (note: "segement" typo) in OutputSettings.
**Python behavior:** Column named `min_ad_segement_separation_seconds`.
**Rust behavior:** Same column name in database for migration compatibility. Rust struct field uses the correct spelling `min_ad_segment_separation_seconds` with a `#[sqlx(rename)]` attribute.
**Rationale:** Migration compatibility. The Rust API will use the correct spelling externally.

## DECISION-007: AMD64-first Docker builds
**Date:** 2026-03-13
**Category:** architecture
**Context:** whisper.cpp cross-compilation for ARM64 is complex due to SIMD/NEON optimizations and cmake toolchain requirements.
**Decision:** Target AMD64 first. ARM64 is a stretch goal.
**Python behavior:** Multi-arch Docker builds (AMD64 + ARM64).
**Rust behavior:** AMD64 only initially.
**Rationale:** Get a working build first, then tackle cross-compilation complexity.

## DECISION-008: Session-based auth with tower-sessions
**Date:** 2026-03-13
**Category:** architecture
**Context:** Python uses Flask session cookies for auth. Need equivalent in Rust.
**Decision:** Use `tower-sessions` with `tower-sessions-sqlx-store` (SQLite-backed sessions). Sessions stored in the same SQLite database.
**Python behavior:** Flask session cookies with server-side session storage.
**Rust behavior:** tower-sessions with SQLite session store.
**Rationale:** Closest match to Python behavior. SQLite store keeps everything in one database. No need for Redis or in-memory sessions.

## DECISION-009: sqlx compile-time checked queries deferred
**Date:** 2026-03-13
**Category:** architecture
**Context:** sqlx supports compile-time query checking via `sqlx::query!()` macro, but requires a live database at compile time.
**Decision:** Start with runtime-checked `sqlx::query_as()` and `sqlx::query()`. Migrate to compile-time checked queries later once the schema stabilizes.
**Rationale:** Faster iteration during initial development. Compile-time checking requires `DATABASE_URL` set during builds and a prepared database, which adds friction during rapid development.

## DECISION-010: Billing/Stripe routes deferred
**Date:** 2026-03-13
**Category:** deferred
**Context:** Python has Stripe billing integration (billing_routes.py).
**Decision:** Defer Stripe billing routes to a later phase. Stub them as 501 Not Implemented.
**Python behavior:** Full Stripe integration with checkout sessions, webhooks, subscription management.
**Rust behavior:** 501 Not Implemented for all billing endpoints.
**Rationale:** Billing is a complex, self-contained feature. Core podcast processing pipeline is higher priority. Can be added after the core is working.

## DECISION-011: Discord OAuth routes deferred
**Date:** 2026-03-13
**Category:** deferred
**Context:** Python has Discord SSO integration (discord_routes.py).
**Decision:** Defer Discord OAuth to a later phase. Stub as 501.
**Python behavior:** Full Discord OAuth flow with guild membership checks.
**Rust behavior:** 501 Not Implemented. Discord status endpoint returns `{"enabled": false}`.
**Rationale:** Same as billing — self-contained feature, lower priority than core pipeline.

## DECISION-012: Local whisper behind feature flag
**Date:** 2026-03-13
**Category:** architecture
**Context:** `whisper-rs` requires cmake, clang, and a C++ toolchain to compile (it builds whisper.cpp from source). This makes the default build heavy and slow.
**Decision:** Put `whisper-rs` behind a cargo feature flag `local-whisper`. Default build only includes remote API transcription backends.
**Python behavior:** Local whisper always available (lazy import of `openai-whisper` Python package).
**Rust behavior:** Local whisper available only when built with `cargo build --features local-whisper`. Docker build always enables it.
**Rationale:** Faster dev iteration without C++ compilation. Docker build (the production path) always enables the feature.

## DECISION-013: sqlx runtime queries instead of compile-time macros
**Date:** 2026-03-13
**Category:** architecture
**Context:** sqlx supports `query!()` compile-time checked macros but requires a live database at compile time.
**Decision:** Use `sqlx::query_as()` with runtime checking. Migrate to compile-time macros once schema stabilizes.
**Rationale:** Faster iteration. Avoids need for `DATABASE_URL` during builds.

## DECISION-014: Legacy .mp3 routes changed
**Date:** 2026-03-13
**Category:** api
**Context:** Python has routes `/post/<guid>.mp3` and `/post/<guid>/original.mp3`. Axum's router doesn't allow path parameters and literal text in the same segment.
**Python behavior:** `/post/abc123.mp3` serves processed audio.
**Rust behavior:** `/post/abc123/mp3` serves processed audio (no dot). `/post/abc123/original.mp3` is unchanged (extension is in its own segment).
**Rationale:** Axum router limitation. RSS feed generator will use the new URL format. Existing podcast clients following RSS will get correct URLs.

## DECISION-015: Password hashing changed to Argon2id
**Date:** 2026-03-13
**Category:** behavior
**Context:** Python backend used bcrypt for password hashing.
**Decision:** Use `argon2` crate with Argon2id algorithm.
**Python behavior:** bcrypt hashing via Flask-Bcrypt.
**Rust behavior:** Argon2id hashing via `argon2` crate.
**Rationale:** Argon2id is the recommended modern password hashing algorithm (OWASP). Migration tool preserves old bcrypt hashes but warns that affected users will need password resets.

## DECISION-016: In-memory session store (temporary)
**Date:** 2026-03-13
**Category:** architecture
**Context:** `tower-sessions-sqlx-store 0.14` depends on `tower-sessions-core 0.13`, but `tower-sessions 0.14` depends on `tower-sessions-core 0.14`. This version mismatch prevents using SQLite-backed sessions.
**Decision:** Use `MemoryStore` from tower-sessions. Sessions are lost on server restart.
**Python behavior:** Server-side sessions persisted across restarts.
**Rust behavior:** Sessions lost on restart until tower-sessions ecosystem versions align.
**Rationale:** Unblocks development. The fix is simply upgrading tower-sessions-sqlx-store once a compatible version is released. TODO tracked in main.rs.

## DECISION-017: Extension-based auth extraction in handlers
**Date:** 2026-03-13
**Category:** architecture
**Context:** Axum handlers that consume `Request<Body>` cannot also extract `Json<T>` (both consume the body). Need a way to access the authenticated user AND parse JSON bodies.
**Decision:** Auth middleware inserts `AuthenticatedUser` into request extensions. Handlers extract via `Option<Extension<AuthenticatedUser>>`. Helper functions `require_admin_user()` and `get_auth_user()` simplify common patterns.
**Python behavior:** `current_user` from Flask-Login, available globally.
**Rust behavior:** `Option<Extension<AuthenticatedUser>>` extractor, with helper functions for admin checks.
**Rationale:** Idiomatic Axum pattern. Extensions are set by middleware and available to all downstream handlers without consuming the body.

## DECISION-018: Simple tokio::spawn scheduler instead of cron
**Date:** 2026-03-13
**Category:** architecture
**Context:** Python used APScheduler for periodic tasks. Rust options include `tokio-cron-scheduler` or a simple tokio loop.
**Decision:** Use a simple `tokio::spawn` loop that reads the refresh interval from the database and sleeps between runs.
**Python behavior:** APScheduler with cron-like syntax.
**Rust behavior:** `tokio::spawn` loop with configurable interval from `app_settings.background_update_interval_minute`.
**Rationale:** Simpler, fewer dependencies. The scheduler only has one task (feed refresh + job enqueue). A full cron library is overkill. The interval is dynamically read from the database each cycle.

## DECISION-019: Legacy migration preserves IDs and schema
**Date:** 2026-03-13
**Category:** migration
**Context:** Need to migrate data from Python SQLite database to Rust backend database.
**Decision:** Direct row-by-row copy preserving all original IDs. Settings tables use column intersection (only copy columns that exist in both schemas). bcrypt password hashes are preserved but flagged as requiring reset.
**Python behavior:** N/A (source).
**Rust behavior:** `migrate_legacy` binary reads old DB (read-only) and writes to new DB with `INSERT OR IGNORE`.
**Rationale:** Preserving IDs maintains foreign key integrity. `INSERT OR IGNORE` makes migration idempotent (safe to re-run). Column intersection handles minor schema differences between Python and Rust settings tables.

## DECISION-020: LLM classification uses raw HTTP instead of litellm
**Date:** 2026-03-14
**Category:** architecture
**Context:** Python uses `litellm` for LLM calls, which abstracts OpenAI, Anthropic, Groq, etc. Rust has no litellm equivalent.
**Decision:** Use raw HTTP calls to OpenAI-compatible `/chat/completions` endpoints via `reqwest`. The `openai_base_url` setting redirects to any compatible provider (Groq, Ollama, etc.).
**Python behavior:** `litellm.completion()` with automatic provider detection from model name prefix.
**Rust behavior:** Direct HTTP POST to `{base_url}/chat/completions` with Bearer auth.
**Rationale:** Most LLM providers support the OpenAI chat completions API format. Using raw HTTP avoids adding a large dependency. The `openai_base_url` config handles provider switching.

## DECISION-021: System prompt embedded via include_str!
**Date:** 2026-03-14
**Category:** architecture
**Context:** Python loads prompts from files at runtime. Rust can embed files at compile time.
**Decision:** System prompt embedded via `include_str!("../../prompts/system_prompt.txt")`. Boundary refinement prompt built in Rust code (not from template) since the original uses Jinja conditionals.
**Python behavior:** Runtime file reads + Jinja2 template rendering.
**Rust behavior:** Compile-time embedding for system prompt; hand-built format strings for templates with conditionals.
**Rationale:** No Jinja2 dependency needed. Compile-time embedding ensures prompts can't be missing at runtime. Boundary refinement prompt is short enough to build in code.

## DECISION-022: Podcast Index API uses SHA1 (not SHA256)
**Date:** 2026-03-14
**Category:** bugfix
**Context:** The initial `sha1_hex` function incorrectly used SHA256. Podcast Index API requires SHA1 for auth header.
**Decision:** Added `sha1` crate and corrected the hash function.
**Rationale:** API compatibility requirement.

## DECISION-023: Discord OAuth — full implementation with DB + env settings
**Date:** 2026-03-14
**Category:** auth
**Context:** Python backend has full Discord OAuth2 flow with guild membership checks and user upsert.
**Decision:** Implemented full OAuth2 authorization code flow using raw reqwest HTTP calls. Settings load from DB (discord_settings table) with env var overrides (DISCORD_CLIENT_ID, etc.). Guild membership check optional. Users upserted by discord_id with collision-free username generation. CSRF state stored in session.
**Python behavior:** Uses `discord_service.py` with `httpx` for API calls.
**Rust behavior:** `api/discord.rs` with `reqwest` and `form_urlencoded`. Callback redirects to `/?error=...` on failure, `/` on success.
**Rationale:** Feature parity with Python. Raw HTTP keeps dependencies minimal.

## DECISION-024: Stripe billing — full implementation with raw HTTP API
**Date:** 2026-03-14
**Category:** billing
**Context:** Python backend uses `stripe` SDK for checkout sessions, portal sessions, webhook handling, and subscription management.
**Decision:** Implemented full Stripe integration using raw reqwest HTTP calls to Stripe REST API with Basic Auth. Pay-What-You-Want model with custom amounts. Webhook signature verification via HMAC-SHA256. Handles subscription lifecycle events (created/updated/deleted/paused/checkout.completed).
**Python behavior:** Uses `stripe` Python SDK.
**Rust behavior:** `api/billing.rs` with `reqwest` + `hmac`/`sha2` for webhook verification. Config via STRIPE_SECRET_KEY, STRIPE_PRODUCT_ID, STRIPE_WEBHOOK_SECRET env vars.
**Rationale:** Raw HTTP avoids adding a Stripe SDK dependency. Stripe's REST API is straightforward with form-encoded params and Basic Auth.

## DECISION-025: One-shot classifier — single-pass LLM ad detection
**Date:** 2026-03-14
**Category:** classification
**Context:** Python has two classification strategies: chunked (multi-call with cue detection) and one-shot (single/few-call with CSV transcript).
**Decision:** Implemented one-shot classifier as `classification/oneshot.rs`. Sends entire transcript as CSV in one LLM call, or splits into 2-hour chunks with 15-minute overlap for long episodes. Returns timestamp-based ad segments with confidence gradients. Model configurable via `oneshot_model` setting. Pipeline dispatches based on feed/app `ad_detection_strategy` field ("llm" vs "oneshot").
**Python behavior:** `oneshot_classifier.py` with litellm.
**Rust behavior:** `classification/oneshot.rs` with raw reqwest to OpenAI-compatible API. Skips boundary refinement step since one-shot returns precise timestamps.
**Rationale:** Feature parity. One-shot is simpler and often better for models with large context windows.
