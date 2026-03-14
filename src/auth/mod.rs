pub mod feed_tokens;
pub mod middleware;
pub mod rate_limiter;

use argon2::password_hash::rand_core::OsRng;
use argon2::password_hash::SaltString;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};

/// Authenticated user info stored in request extensions by auth middleware.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AuthenticatedUser {
    pub id: i64,
    pub username: String,
    pub role: String,
}

pub fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2.hash_password(password.as_bytes(), &salt)?;
    Ok(hash.to_string())
}

/// Result of password verification — indicates whether a re-hash is needed.
pub enum VerifyResult {
    /// Password matched (no re-hash needed).
    Ok,
    /// Password matched a legacy bcrypt hash — caller should re-hash with Argon2.
    OkNeedsRehash,
    /// Password did not match.
    Failed,
}

pub fn verify_password(password: &str, hash: &str) -> VerifyResult {
    // Try Argon2 first (current standard)
    if let Ok(parsed_hash) = PasswordHash::new(hash) {
        if Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok()
        {
            return VerifyResult::Ok;
        }
    }

    // Fall back to bcrypt for migrated hashes
    if hash.starts_with("$2b$") || hash.starts_with("$2a$") || hash.starts_with("$2y$") {
        if bcrypt::verify(password, hash).unwrap_or(false) {
            return VerifyResult::OkNeedsRehash;
        }
    }

    VerifyResult::Failed
}

/// Validate password strength (matches Python: min 6 chars).
pub fn validate_password(password: &str) -> Result<(), String> {
    if password.len() < 6 {
        return Err("Password must be at least 6 characters.".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_and_verify_argon2() {
        let password = "test_password_123";
        let hash = hash_password(password).unwrap();
        assert!(matches!(verify_password(password, &hash), VerifyResult::Ok));
        assert!(matches!(verify_password("wrong_password", &hash), VerifyResult::Failed));
    }

    #[test]
    fn test_verify_bcrypt_fallback() {
        let password = "test_password_123";
        let bcrypt_hash = bcrypt::hash(password, bcrypt::DEFAULT_COST).unwrap();
        assert!(matches!(verify_password(password, &bcrypt_hash), VerifyResult::OkNeedsRehash));
        assert!(matches!(verify_password("wrong", &bcrypt_hash), VerifyResult::Failed));
    }

    #[test]
    fn test_validate_password() {
        assert!(validate_password("short").is_err());
        assert!(validate_password("longenough").is_ok());
    }
}
