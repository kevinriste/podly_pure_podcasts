# Podly Rust Migration: Engineering Assessment

**Date:** March 15, 2026
**Scope:** Complete backend rewrite from Python/Flask to Rust/Axum
**Branch:** `rust-backend` (90 commits, 11,884 lines of Rust)

---

## Executive Summary

This is a full backend rewrite of a podcast ad-removal application. The Rust version achieves API parity with the Python original across all endpoints (auth, config, feeds, posts, billing, Discord OAuth, RSS generation, processing pipeline). It uses 19x less memory at steady state and produces a 14x smaller Docker image.

The rewrite is **not yet production-ready**. It has never processed a real podcast episode end-to-end, has essentially zero test coverage, and several architectural decisions trade long-term maintainability for development speed. These are addressable issues, not fundamental problems.

**Important note on scope:** The Rust rewrite includes features from unmerged Python feature branches — notably the oneshot classifier (`kevin/feat/oneshot-llm-strategy`), 12-factor env var precedence (`kevin/feat/env-vars-authoritative`), and chapter-based ad detection (in `origin/preview` but not `origin/main`). See [NON_UPSTREAM_FEATURES.md](NON_UPSTREAM_FEATURES.md) for the full breakdown.

---

## Is This Migration a Good Idea?

**Yes, with caveats.**

The Python backend runs two processes consuming 1.4 GiB of RAM to serve what is essentially a CRUD API with background job processing. The Rust version does the same work in 82 MiB. For a self-hosted application running on resource-constrained hardware alongside other services, this is a meaningful difference.

The more significant benefit is operational: a single statically-linked binary with no Python runtime, no pip dependency tree, no Gunicorn worker management. Deployment is "copy one file." The Docker image drops from 8.66 GB to 603 MB, which matters for CI build times and ARM devices.

The cost is maintainability. The Python codebase is more approachable to contributors. Rust's learning curve is real. The rewrite was done by AI in three days; a human maintaining it needs to be comfortable with async Rust, raw SQL, and the Axum middleware stack.

**Recommendation:** Proceed, but invest in test coverage and end-to-end validation before merging. The performance gains are real and the code is structurally sound, but it has not been proven under real workloads.

---

## What You're Getting

### Complete API Parity
Every endpoint from the Python backend has a Rust equivalent with matching:
- HTTP status codes and error messages
- JSON response shapes and field names
- Authentication and authorization logic
- Edge case handling (admin-only routes, disabled-auth mode, feed #1 special case)

Verified through 8 independent audit cycles with specialized agents covering auth, billing, jobs, config, feeds, posts, DB queries, and the processing pipeline.

### Full Processing Pipeline
Download → Transcribe → Classify → Refine → Merge → Cut audio. Three transcription backends (remote Whisper API, Groq, local whisper.cpp). Two classification strategies (chunked LLM, one-shot LLM). Boundary refinement with LLM + heuristic fallback. Ad segment merging with configurable thresholds.

### Feature-Complete Integrations
- **Stripe billing:** Full checkout, portal, webhook, subscription lifecycle
- **Discord OAuth:** Login flow, guild membership checks, user upsert
- **RSS feed generation:** Token-authenticated feeds, aggregate user feeds
- **Background scheduler:** Periodic feed refresh and job processing

### Data Migration Tool
Binary that copies all 12 tables from a Python-era SQLite database to the Rust schema. Preserves IDs, handles schema differences, supports dry-run mode.

---

## Architectural Decisions and Tradeoffs

### Why Not Use the Stripe SDK?

There is a community Rust crate (`stripe-rs`, ~4k downloads/month). The rewrite uses raw HTTP instead.

**What raw HTTP looks like in practice:** 6 helper functions totaling ~120 lines, calling Stripe's REST API with `reqwest::Client` and Basic Auth. Webhook signature verification is 35 lines of manual HMAC-SHA256.

**What you lose:** Type-safe request/response structs, automatic retry logic, idempotency key support, and forward compatibility with Stripe API changes. If Stripe deprecates a parameter name or changes a response shape, the Rust code breaks silently at runtime.

**What you gain:** No additional dependency (~3k lines of generated code in stripe-rs), simpler error handling for the 6 endpoints actually used.

**Assessment:** Acceptable for the current scope (pay-what-you-want model with simple subscription lifecycle). Would become a maintenance liability if billing complexity grows. Worth revisiting if Stripe integration expands.

### Why sqlx Instead of an ORM?

The rewrite uses `sqlx` with hand-written SQL strings — 88 query functions across 911 lines. No compile-time query checking (would require a live database during builds).

**What you lose compared to Diesel or SeaORM:**
- No compile-time schema validation. If a column is renamed or removed, it's a runtime error.
- No relationship traversal. Joins are written manually.
- Refactoring friction. Renaming a column means grep-and-replace across SQL strings.

**What you gain:**
- Explicit control over every query. No hidden N+1 problems, no ORM-generated suboptimal SQL.
- Easy to port from the Python codebase, which also used explicit SQL through SQLAlchemy.
- SQLite-specific features used directly (e.g., `INSERT OR IGNORE`, `CAST`).

**Assessment:** Reasonable for a codebase of this size. The risk is manageable because the schema is stable (inherited from the Python app) and the query count is finite. The `sqlx::query!()` compile-time macro could be adopted later without rewriting queries — it's the same SQL strings, just checked at build time.

### Why PodcastIndex Instead of iTunes?

The Python backend used the iTunes Search API for podcast discovery. The Rust version uses PodcastIndex.

**Why the switch:** PodcastIndex requires API key authentication (SHA1 signature), which was already implemented. iTunes is unauthenticated but has been unreliable and is technically undocumented. PodcastIndex is an open, community-maintained index.

**Impact:** The response format is normalized to the same shape (`{title, feedUrl, description, author, artworkUrl}`), so the frontend sees no difference. The catalog coverage is comparable for mainstream podcasts. Niche or region-specific podcasts may have different availability between the two indexes.

**Assessment:** Neutral change. Could add iTunes as a fallback if catalog gaps appear.

### LLM Integration: genai Crate vs litellm

Python uses `litellm`, which abstracts 100+ LLM providers behind a single `completion()` call. Rust uses the `genai` crate.

**genai** natively supports OpenAI, Anthropic, Gemini, and Groq. It handles model name routing (e.g., `anthropic::claude-3-haiku`) and structured outputs. It does not support the long tail of providers that litellm covers (Azure OpenAI, Bedrock, Vertex AI, Cohere, etc.).

**Practical impact:** If the user's `LLM_MODEL` is set to a provider genai doesn't support, classification will fail. This is a regression from litellm's broader compatibility.

**Assessment:** Adequate for the common case (OpenAI, Anthropic, Gemini, Groq cover >95% of users). Document the supported providers clearly. Consider adding an OpenAI-compatible HTTP fallback for unsupported providers.

### Password Hashing: Argon2id with bcrypt Fallback

The Python backend used bcrypt. The Rust version uses Argon2id (OWASP-recommended) with automatic detection and transparent re-hashing of legacy bcrypt passwords on login.

**Assessment:** Strictly better. No user-facing impact — existing passwords work, new passwords get stronger hashing.

---

## What's Incomplete or Risky

### No Test Coverage (HIGH RISK)

The Rust codebase has 3 unit tests, all in the auth module (password hashing). Every other module — billing, feeds, transcription, pipeline, config, queries — has zero tests.

This means:
- Schema changes break silently until runtime.
- API parity with Python is validated by audit agents reading code, not by automated tests.
- The processing pipeline has never been tested with real audio.
- Error handling paths are unverified.

**This is the single biggest risk.** The code reads correctly, but "reads correctly" is not the same as "works correctly."

### Never Tested End-to-End (HIGH RISK)

No podcast episode has ever been processed through the Rust pipeline. The entire flow — download audio, transcribe via Whisper API, classify ads via LLM, refine boundaries, merge segments, cut audio with ffmpeg — is untested as an integrated system.

The Python backend has been processing real podcasts for months. The Rust backend has been verified through code audits and endpoint-level response comparison, but never under real workload.

### Local Whisper and ARM64 Docker (INCOMPLETE)

Local whisper (via `whisper-rs` / whisper.cpp) is **implemented in code** behind a feature flag (`local-whisper`). However:

- The Dockerfile does not include the C++ build toolchain or whisper model files needed to enable it.
- ARM64 Docker builds are not configured. `whisper-rs-sys` requires architecture-specific SIMD compilation.
- The current Docker image is AMD64-only.

If local transcription or ARM64 deployment is required, this needs explicit build pipeline work.

### Ad Segment Merging is Simplified (LOW RISK)

Both backends implement multi-pass ad merging: proximity grouping, content-aware keyword extraction (URLs, promo codes, brand names, phone numbers), weak-group filtering, short-segment removal, and last-segment extension. The Rust implementation queries transcript text from the database for each ad group and runs the same regex patterns as Python's `AdMerger`.

### No Data Migration Needed (CORRECTED)

The Rust binary can use the existing Python SQLite database file directly. SQLite column types (`DateTime` vs `TEXT`, `JSON` vs `TEXT`) are just type affinities — the stored bytes are identical between the two backends.

The Rust schema adds a few columns that don't exist in upstream Python (`oneshot_model`, `oneshot_max_chunk_duration_seconds`, etc.). These are from unmerged feature branches, not schema incompatibilities. They are added automatically via `ALTER TABLE ... ADD COLUMN` on startup.

A `migrate_legacy` binary exists for copying data between separate database files, but it is unnecessary for the standard deployment path. Just point the Rust binary at the same `.db` file the Python backend uses.

### Error Context is Sometimes Lost (LOW RISK)

Several error paths convert `sqlx::Error` to `String` early, losing the structured error information. This makes debugging database issues harder in production logs. Not a correctness issue, but an operational friction point.

### A Few unwrap() Calls in Production Code (LOW RISK)

Three `.unwrap()` calls exist in hot paths (HTTP response builders, vector operations). These are safe in context (the values are guaranteed to exist) but violate defensive coding practices. A panic in an Axum handler crashes the request, not the server, so the blast radius is limited.

---

## What Must Be Addressed Before Merging

### Must-Have
1. **End-to-end processing test.** Process at least one real podcast episode through the Rust pipeline and verify the output audio matches expectations.
2. **Integration test for critical paths.** At minimum: login flow, feed add/refresh, post process/reprocess, config read/write.
3. **Docker image with local whisper** (if local transcription is a requirement).

### Should-Have
4. **Compile-time query checking.** Switch from `sqlx::query_as()` to `sqlx::query_as!()` for the most critical queries. This catches schema mismatches at build time.
5. **ARM64 Docker build** (if ARM deployment is needed).
6. **Stripe SDK evaluation.** Assess whether `stripe-rs` is mature enough to replace raw HTTP, reducing Stripe maintenance burden.

### Nice-to-Have
7. Token rate limiting for LLM API calls.
9. `max_completion_tokens` support for newer OpenAI models.
10. Input token validation to prevent oversized prompts.

---

## Decisions Worth Reconsidering

| Decision | Current Choice | Concern | Recommendation |
|----------|---------------|---------|----------------|
| No ORM | sqlx with raw SQL | Schema changes break at runtime | Adopt `sqlx::query!()` compile-time checking |
| No Stripe SDK | Raw reqwest HTTP | Maintenance burden grows with complexity | Evaluate `stripe-rs` for type safety |
| PodcastIndex only | Replaced iTunes | Potential catalog gaps | Add iTunes as fallback option |
| genai crate | Replaced litellm | Fewer supported LLM providers | Document supported providers; add HTTP fallback |
| No tests | 3 unit tests total | Highest risk area | Invest in integration test suite |
| AMD64 only | No ARM64 Docker | Limits deployment targets | Add cross-compilation CI step |

---

## Performance Data

All measurements at steady state on production server (23.3 GiB RAM, Linux 6.8), both backends serving the same 25 GB podcast library with 311 MB SQLite database.

| Metric | Python | Rust | Ratio |
|--------|--------|------|-------|
| Memory (RSS) | 1,412 MiB (2 processes) | 82 MiB (1 process) | 17.2x |
| Docker image | 8.66 GB | 603 MB | 14.4x |
| Block I/O written | 3.1 GB (6 days) | 63 MB (1 hour) | ~49x |
| CPU (idle) | 0.01% | 0.00% | Both negligible |
| Processes | 2 (main + writer) | 1 (async binary) | - |
| Threads | 14 | 9 | - |
| Binary size | Full Python 3.12 + deps | 21 MB | - |

**Note on I/O comparison:** The I/O figures are not directly comparable (different uptime periods). Python's higher I/O reflects ORM journaling overhead and the separate writer process over 6 days of active use.

**Note on processing throughput:** Backend language has minimal impact on actual podcast processing time. The bottleneck is external API calls (Whisper transcription: seconds to minutes, LLM classification: seconds per chunk) and ffmpeg audio processing. Rust's advantage is resource efficiency, not processing speed.

---

## Conclusion

The Rust rewrite is architecturally sound, achieves comprehensive API parity, and delivers real resource efficiency gains. The code has been through 8 audit cycles and 110+ parity fixes.

It is not yet ready to merge. The absence of automated tests and end-to-end validation means the rewrite's correctness is asserted by code review, not proven by execution. For a system that manipulates audio files and makes LLM API calls with real money (Whisper API, LLM tokens), this gap needs closing.

**Recommended path forward:**
1. Run one real episode through the pipeline, verify output.
2. Add integration tests for the critical path (login → add feed → process episode → serve RSS).
3. Address Docker/deployment gaps (local whisper, ARM64) based on deployment requirements.
4. Merge with confidence.

The migration is worth completing. The remaining work is validation and hardening, not feature development.
