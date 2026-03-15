# Non-Upstream Features in Rust Rewrite

The Rust rewrite includes features that are NOT in the upstream Python codebase (`origin/main`). These were pulled from unmerged local feature branches. They should be isolated into separate commits so the core rewrite can be reviewed and merged independently.

## Feature Origin Map

### In upstream `origin/main`
Everything else — auth, config, feeds, posts, billing, Discord, RSS, scheduler, processing pipeline (chunked LLM classification), transcription, audio processing. **This is what the Rust rewrite should target for parity.**

### In `origin/preview` only (not main)
- `Feed.ad_detection_strategy` column — per-feed strategy selection ("llm", "chapter", "inherit")
- Chapter-based ad detection strategy (reads ffprobe chapters, filters by title)
- Various CI/build improvements, datetime fixes, word boundary refiner improvements

### From `kevin/feat/oneshot-llm-strategy` (not merged anywhere upstream)
- **Oneshot classifier** (`src/classification/oneshot.rs`) — single-pass LLM ad detection
- `llm_settings.oneshot_model` column
- `llm_settings.oneshot_max_chunk_duration_seconds` column
- `llm_settings.oneshot_chunk_overlap_seconds` column
- `User.ad_detection_strategy` column (Python version; Rust moved it to `app_settings`)

### From `kevin/feat/env-vars-authoritative` (not merged anywhere upstream)
- 12-factor env var precedence model — env vars override DB config at runtime, never persisted back
- `GET /api/config` returns `env_overrides` metadata with `read_only: true`
- `PUT /api/config` strips env-overridden fields before saving
- `read_only_fields` list in config response

### Invented by the Rust rewrite (no Python equivalent anywhere)
- `app_settings.ad_detection_strategy` column — global default (Python has it per-user on `kevin/feat/oneshot-llm-strategy`, `the-plan.md` proposes moving it to `app_settings`)
- `processing_settings.max_overlap_segments` column — origin unclear, may be from an untracked local change

## Impact on Schema

**No migration tool is needed.** The Rust binary can use the upstream Python SQLite database directly. SQLite column types (`DateTime` vs `TEXT`, `JSON` vs `TEXT`) are just affinities — the actual stored bytes are identical.

The non-upstream features add columns that don't exist in the upstream schema. These should be added via `ALTER TABLE ... ADD COLUMN` on startup (the Rust code already has this pattern in `db/mod.rs` for `max_overlap_segments`).

## Isolation Plan

To produce a clean upstream-parity rewrite, these features should be separated:

### Commit group 1: Core upstream parity
- All API endpoints matching `origin/main` behavior
- Chunked LLM classification only (no oneshot)
- No env var precedence model (config reads/writes DB directly)
- No `ad_detection_strategy` anywhere
- Schema identical to `origin/main`

### Commit group 2: Preview parity (`origin/preview` additions)
- `Feed.ad_detection_strategy` column + chapter-based detection
- Per-feed strategy selection in pipeline

### Commit group 3: Env var precedence (`kevin/feat/env-vars-authoritative`)
- 12-factor config model
- `env_overrides` in config response
- Field stripping on PUT

### Commit group 4: Oneshot classifier (`kevin/feat/oneshot-llm-strategy`)
- Oneshot classifier implementation
- `llm_settings.oneshot_*` columns
- `app_settings.ad_detection_strategy` global default
- Pipeline dispatch for oneshot strategy

## Current State

All four groups are **interleaved throughout the codebase**. Separating them would require careful code extraction, not just git operations. The classification pipeline, config endpoints, and schema all reference these features inline.

For practical purposes, the Rust rewrite can be merged as-is (with all features) and the Python upstream can adopt the features separately. The alternative — stripping features from the Rust code — would be significant rework for limited benefit, since the features work correctly and the Python branches are intended to be merged eventually.
