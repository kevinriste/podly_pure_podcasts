use async_trait::async_trait;
use sqlx::SqlitePool;
use tower_sessions::cookie::time::OffsetDateTime;
use tower_sessions::{
    session::{Id, Record},
    session_store, SessionStore,
};

#[derive(Clone, Debug)]
pub struct SqliteSessionStore {
    pool: SqlitePool,
}

impl SqliteSessionStore {
    pub async fn new(pool: SqlitePool) -> Result<Self, sqlx::Error> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY NOT NULL,
                data TEXT NOT NULL,
                expiry_date INTEGER NOT NULL
            )",
        )
        .execute(&pool)
        .await?;

        Ok(Self { pool })
    }

    fn id_to_string(id: &Id) -> String {
        format!("{}", id.0)
    }

    fn string_to_id(s: &str) -> Option<Id> {
        s.parse::<i128>().ok().map(Id)
    }
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn create(&self, record: &mut Record) -> session_store::Result<()> {
        loop {
            let id_str = Self::id_to_string(&record.id);
            let existing: Option<(String,)> =
                sqlx::query_as("SELECT id FROM sessions WHERE id = ?")
                    .bind(&id_str)
                    .fetch_optional(&self.pool)
                    .await
                    .map_err(|e| session_store::Error::Backend(e.to_string()))?;

            if existing.is_none() {
                break;
            }
            record.id = Id::default();
        }

        self.save(record).await
    }

    async fn save(&self, record: &Record) -> session_store::Result<()> {
        let id_str = Self::id_to_string(&record.id);
        let data = serde_json::to_string(&record.data)
            .map_err(|e| session_store::Error::Encode(e.to_string()))?;
        let expiry = record.expiry_date.unix_timestamp();

        sqlx::query(
            "INSERT INTO sessions (id, data, expiry_date) VALUES (?, ?, ?)
             ON CONFLICT(id) DO UPDATE SET data = excluded.data, expiry_date = excluded.expiry_date",
        )
        .bind(&id_str)
        .bind(&data)
        .bind(expiry)
        .execute(&self.pool)
        .await
        .map_err(|e| session_store::Error::Backend(e.to_string()))?;

        Ok(())
    }

    async fn load(&self, session_id: &Id) -> session_store::Result<Option<Record>> {
        let id_str = Self::id_to_string(session_id);
        let now = OffsetDateTime::now_utc().unix_timestamp();

        let row: Option<(String, String, i64)> = sqlx::query_as(
            "SELECT id, data, expiry_date FROM sessions WHERE id = ? AND expiry_date > ?",
        )
        .bind(&id_str)
        .bind(now)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| session_store::Error::Backend(e.to_string()))?;

        match row {
            Some((id, data, expiry)) => {
                let session_id = Self::string_to_id(&id)
                    .ok_or_else(|| session_store::Error::Decode("invalid session id".into()))?;
                let data = serde_json::from_str(&data)
                    .map_err(|e| session_store::Error::Decode(e.to_string()))?;
                let expiry_date = OffsetDateTime::from_unix_timestamp(expiry)
                    .map_err(|e| session_store::Error::Decode(e.to_string()))?;

                Ok(Some(Record {
                    id: session_id,
                    data,
                    expiry_date,
                }))
            }
            None => Ok(None),
        }
    }

    async fn delete(&self, session_id: &Id) -> session_store::Result<()> {
        let id_str = Self::id_to_string(session_id);

        sqlx::query("DELETE FROM sessions WHERE id = ?")
            .bind(&id_str)
            .execute(&self.pool)
            .await
            .map_err(|e| session_store::Error::Backend(e.to_string()))?;

        Ok(())
    }
}
