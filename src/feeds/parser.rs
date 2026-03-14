use feed_rs::parser;

#[derive(Debug)]
pub struct ParsedFeed {
    pub title: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub image_url: Option<String>,
    pub episodes: Vec<ParsedEpisode>,
}

#[derive(Debug)]
pub struct ParsedEpisode {
    pub guid: String,
    pub title: String,
    pub description: Option<String>,
    pub audio_url: Option<String>,
    pub release_date: Option<chrono::DateTime<chrono::Utc>>,
    pub duration: Option<i64>,
    pub image_url: Option<String>,
}

/// Fetch and parse an RSS feed from a URL.
pub async fn fetch_and_parse(url: &str) -> Result<ParsedFeed, FeedParseError> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| FeedParseError::Fetch(e.to_string()))?;

    let bytes = response
        .bytes()
        .await
        .map_err(|e| FeedParseError::Fetch(e.to_string()))?;

    parse_feed_bytes(&bytes, url)
}

/// Parse feed from raw bytes.
pub fn parse_feed_bytes(bytes: &[u8], _url: &str) -> Result<ParsedFeed, FeedParseError> {
    let feed = parser::parse(bytes).map_err(|e| FeedParseError::Parse(e.to_string()))?;

    let title = feed
        .title
        .map(|t| t.content)
        .unwrap_or_else(|| "Untitled Feed".to_string());

    let description = feed.description.map(|d| d.content);

    let author = feed.authors.first().map(|a| a.name.clone());

    let image_url = feed.logo.map(|l| l.uri).or(feed.icon.map(|i| i.uri));

    let episodes = feed
        .entries
        .into_iter()
        .map(|entry| {
            let guid = entry.id;
            let title = entry
                .title
                .map(|t| t.content)
                .unwrap_or_else(|| "Untitled".to_string());

            let description = entry
                .summary
                .map(|s| s.content)
                .or_else(|| entry.content.and_then(|c| c.body));

            let audio_url = entry
                .media
                .first()
                .and_then(|m| m.content.first())
                .and_then(|c| c.url.as_ref())
                .map(|u| u.to_string())
                .or_else(|| {
                    entry
                        .links
                        .iter()
                        .find(|l| {
                            l.media_type
                                .as_ref()
                                .is_some_and(|mt| mt.starts_with("audio/"))
                        })
                        .map(|l| l.href.clone())
                });

            let release_date = entry.published.or(entry.updated);

            let duration = entry
                .media
                .first()
                .and_then(|m| m.content.first())
                .and_then(|c| c.duration)
                .map(|d| d.as_secs() as i64);

            let image_url = entry
                .media
                .first()
                .and_then(|m| m.thumbnails.first().map(|t| t.image.uri.clone()));

            ParsedEpisode {
                guid,
                title,
                description,
                audio_url,
                release_date,
                duration,
                image_url,
            }
        })
        .collect();

    Ok(ParsedFeed {
        title,
        description,
        author,
        image_url,
        episodes,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum FeedParseError {
    #[error("fetch error: {0}")]
    Fetch(String),
    #[error("parse error: {0}")]
    Parse(String),
}
