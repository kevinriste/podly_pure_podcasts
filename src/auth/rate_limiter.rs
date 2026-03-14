use std::collections::HashMap;
use std::time::{Duration, Instant};

struct FailureState {
    attempts: u32,
    blocked_until: Option<Instant>,
    last_attempt: Instant,
}

/// In-memory exponential backoff rate limiter for authentication failures.
pub struct FailureRateLimiter {
    storage: HashMap<String, FailureState>,
    max_backoff_seconds: u64,
    warm_up_attempts: u32,
}

impl FailureRateLimiter {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
            max_backoff_seconds: 300,
            warm_up_attempts: 3,
        }
    }

    /// Register a failed attempt. Returns backoff seconds (0 if still in warm-up).
    pub fn register_failure(&mut self, key: &str) -> u64 {
        let now = Instant::now();

        let state = self
            .storage
            .entry(key.to_string())
            .or_insert_with(|| FailureState {
                attempts: 0,
                blocked_until: None,
                last_attempt: now,
            });

        state.attempts += 1;
        state.last_attempt = now;

        let backoff_seconds = if state.attempts > self.warm_up_attempts {
            let exponent = state.attempts - self.warm_up_attempts;
            let backoff = 2u64.pow(exponent).min(self.max_backoff_seconds);
            state.blocked_until = Some(now + Duration::from_secs(backoff));
            backoff
        } else {
            state.blocked_until = None;
            0
        };

        self.prune_stale(now);
        backoff_seconds
    }

    pub fn register_success(&mut self, key: &str) {
        self.storage.remove(key);
    }

    /// Returns remaining backoff seconds, or None if not rate-limited.
    pub fn retry_after(&mut self, key: &str) -> Option<u64> {
        let state = self.storage.get(key)?;
        let blocked_until = state.blocked_until?;
        let now = Instant::now();

        if blocked_until <= now {
            self.storage.remove(key);
            return None;
        }

        let remaining = (blocked_until - now).as_secs();
        if remaining == 0 {
            self.storage.remove(key);
            return None;
        }

        Some(remaining)
    }

    fn prune_stale(&mut self, now: Instant) {
        let stale_keys: Vec<String> = self
            .storage
            .iter()
            .filter(|(_, state)| now.duration_since(state.last_attempt) > Duration::from_secs(3600))
            .map(|(key, _)| key.clone())
            .collect();

        for key in stale_keys {
            self.storage.remove(&key);
        }
    }
}

impl Default for FailureRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}
