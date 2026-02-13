# Plan to implement

**Plan:** Align oneshot with global settings pattern + add retries  

---

## Context

The oneshot classifier has its own `OneShotConfig` that's never hydrated from the DB (only Pydantic defaults), and `ad_detection_strategy` default lives on the `User` model — a pattern used nowhere else.

The `auto_whitelist` pattern is: global default in `AppSettings` → per-feed override on `Feed`.

We need to match that, remove the per-user paradigm, and make oneshot respect the same global LLM settings (timeout, retries) that chunked does.

---

## Changes

### 1. Models (`src/app/models.py`)

- `LLMSettings`: Add `oneshot_model` column (`String(100)`, nullable=True)  
- `AppSettings`: Add `ad_detection_strategy` column (`String(20)`, NOT NULL, default="llm")  
- `User`: Remove `ad_detection_strategy` and `oneshot_model` columns  

---

### 2. Defaults (`src/shared/defaults.py`)

- Remove `ONESHOT_TIMEOUT_SEC` (oneshot will use `OPENAI_DEFAULT_TIMEOUT_SEC`)  
- Remove `ONESHOT_DEFAULT_MODEL`, `ONESHOT_MAX_CHUNK_DURATION_SECONDS`, `ONESHOT_CHUNK_OVERLAP_SECONDS` as constants

---

### 3. Pydantic config (`src/shared/config.py`)

- Remove `OneShotConfig` class  
- Remove `oneshot: OneShotConfig` field from `Config`  
- Add `oneshot_model: str to `Config`, see below for default value strategy (inserted by migration and user-changeable thereafter)
- Add `ad_detection_strategy: str to `Config`, see below for default value strategy (inserted by migration and user-changeable thereafter)

---

### 4. Config store (`src/app/config_store.py`)

- `read_combined()` llm section (~line 443): Add `"oneshot_model": llm.oneshot_model`  
- `read_combined()` app section (~line 467): Add `"ad_detection_strategy": app_s.ad_detection_strategy`  
- `_update_section_llm()` (~line 482): Add `"oneshot_model"` to key list  
- `_update_section_app()` (~line 601): Add `"ad_detection_strategy"` to key list  
- `to_pydantic_config()` (~line 719):  
  - Map `data["llm"]["oneshot_model"]` → `oneshot_model`
  - Map `data["app"]["ad_detection_strategy"]` → `ad_detection_strategy`  
- `ensure_defaults()`: The existing pattern auto-creates rows with column defaults, so no change needed  

---

### 5. Oneshot classifier (`src/podcast_processor/oneshot_classifier.py`)

- `self.config.oneshot.model` → `self.config.oneshot_model`  
- `self.config.oneshot.timeout_sec` → `self.config.openai_timeout`  
- Add retry loop in `_call_llm()` using `self.config.llm_max_retry_attempts`  
  - Wrap the `litellm.completion()` call in a `for` loop  
  - Use exponential backoff  
  - Catch exceptions  
  - Update `model_call.retry_attempts`  
  - Basically just copy how retry attempts are happening in the other processor

---

### 6. Podcast processor (`src/podcast_processor/podcast_processor.py`)

- Strategy resolution (~line 163):  
  - Replace user-level fetch with `self.config.ad_detection_strategy` as global default  
  - Keep feed-level override (`!= "inherit"` wins)  
- `self.config.oneshot.model` → `self.config.oneshot_model` (~line 478)  
- Remove `cached_user_strategy`, `cached_user_oneshot_model` variables  

---

### 7. Auth routes (`src/app/routes/auth_routes.py`)

- Remove `_apply_ad_strategy_updates()` helper function  
- Remove `ad_detection_strategy` / `oneshot_model` from user list serialization (~line 214)  
- Remove ad strategy / oneshot handling from `update_user_route`  
- Remove `writer_client` import if no longer used  

---

### 8. Frontend types + API

- `frontend/src/types/index.ts`  
  - Remove `ad_detection_strategy` / `oneshot_model` from `ManagedUser`  
  - Add `oneshot_model?: string | null` to `LLMConfig`  
  - Add `ad_detection_strategy: string` to `AppConfigUI`  

- `frontend/src/services/api.ts`  
  - Remove `ad_detection_strategy` / `oneshot_model` from `updateUser` payload  
  - Remove from `listUsers` return type  

---

### 9. User management tab (`frontend/src/components/config/tabs/UserManagementTab.tsx`)

- Remove `handleStrategyChange` and `handleOneshotModelChange` handlers  
- Remove per-user strategy dropdown and oneshot model input from user cards  

---

### 10. Settings sections (Advanced tab)

- `frontend/src/components/config/sections/LLMSection.tsx`:  
  - Add `oneshot_model` text input field  

- `frontend/src/components/config/sections/AppSection.tsx`:  
  - Add `ad_detection_strategy` dropdown ("LLM (chunked)" / "One-shot LLM")  

---

### 11. Feed settings modal (`frontend/src/components/FeedSettingsModal.tsx`)

- Change "Use user default" label → "Use global default"  
- Update inherit description to reference global setting instead of user  
- Add to oneshot description:  
  - "Boundary and word-level refinement do not apply to one-shot processing."  

---

### 12. Migration (manually written, validated with SQLite simulation)

- Add `oneshot_model` to `llm_settings` (nullable)  
- Add `ad_detection_strategy` to `app_settings` (NOT NULL, default "llm")  
- Data migration: copy first non-default user `ad_detection_strategy` value to `app_settings`  
- Drop `ad_detection_strategy` and `oneshot_model` from `users` (via `batch_alter_table` for SQLite)  
- Downgrade: reverse the column moves  

### 13. Additional notes

- Point out any situations where we have created settings that cannot be changed in the UI. We will need to plan to fix these.
- There should be no hidden defaults; if there's a default it needs to be auto-populated in the database by a migration and show up in the settings UI so the user can change it in the database going forward
- Create a plan to flatten out migrations once they're all committed, but don't perform it. Should really only need one for this whole new feature-set since nothing's been commited to the main repo yet
- Add option to also regenerate transcript (not set by default) when reprocessing in UI (unless this was already addressed)

---

## Verification

- `npx tsc --noEmit` — frontend type checking  
- `npx vite build` — frontend builds  
- `black` on changed Python files  
- `pylint` on changed Python files (no new warnings)  
- SQLite migration simulation (upgrade + downgrade)  
