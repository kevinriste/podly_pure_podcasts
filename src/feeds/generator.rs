use chrono::Utc;
use quick_xml::Writer;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use std::io::Cursor;

use crate::db::models::{Feed, Post};

/// Generate an RSS 2.0 feed XML string for a single feed with its posts.
pub fn generate_rss_feed(
    feed: &Feed,
    posts: &[Post],
    base_url: &str,
    token_suffix: &str,
) -> Result<String, quick_xml::Error> {
    let title = format!("[podly] {}", feed.title);
    let link = format!("{}/feed/{}{}", base_url, feed.id, token_suffix);
    let description = feed.description.as_deref().unwrap_or("");
    let image_url = feed.image_url.as_deref();

    generate_rss_xml(
        &title,
        &link,
        description,
        image_url,
        posts,
        base_url,
        token_suffix,
        false,
    )
}

/// Generate an RSS 2.0 feed XML string for an aggregate feed.
pub fn generate_aggregate_rss_feed(
    title: &str,
    description: &str,
    user_id: i64,
    posts: &[Post],
    base_url: &str,
    token_suffix: &str,
) -> Result<String, quick_xml::Error> {
    let link = format!("{}/feed/user/{}{}", base_url, user_id, token_suffix);
    let image_url = Some(format!(
        "{}/static/images/logos/manifest-icon-512.maskable.png",
        base_url
    ));

    generate_rss_xml(
        title,
        &link,
        description,
        image_url.as_deref(),
        posts,
        base_url,
        token_suffix,
        true,
    )
}

fn generate_rss_xml(
    title: &str,
    link: &str,
    description: &str,
    image_url: Option<&str>,
    posts: &[Post],
    base_url: &str,
    token_suffix: &str,
    prepend_feed_title: bool,
) -> Result<String, quick_xml::Error> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    // <rss>
    let mut rss = BytesStart::new("rss");
    rss.push_attribute(("version", "2.0"));
    rss.push_attribute(("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd"));
    rss.push_attribute((
        "xmlns:content",
        "http://purl.org/rss/1.0/modules/content/",
    ));
    rss.push_attribute(("xmlns:atom", "http://www.w3.org/2005/Atom"));
    writer.write_event(Event::Start(rss))?;

    // <channel>
    writer.write_event(Event::Start(BytesStart::new("channel")))?;

    write_text_element(&mut writer, "title", title)?;
    write_text_element(&mut writer, "description", description)?;
    write_text_element(&mut writer, "link", link)?;

    // lastBuildDate
    let last_build_date = Utc::now().format("%a, %d %b %Y %H:%M:%S GMT").to_string();
    write_text_element(&mut writer, "lastBuildDate", &last_build_date)?;

    // Full <image> element
    if let Some(img_url) = image_url {
        writer.write_event(Event::Start(BytesStart::new("image")))?;
        write_text_element(&mut writer, "url", img_url)?;
        write_text_element(&mut writer, "title", title)?;
        write_text_element(&mut writer, "link", link)?;
        writer.write_event(Event::End(BytesEnd::new("image")))?;

        let mut itunes_image = BytesStart::new("itunes:image");
        itunes_image.push_attribute(("href", img_url));
        writer.write_event(Event::Empty(itunes_image))?;
    }

    // Self-referencing atom link
    let mut atom_link = BytesStart::new("atom:link");
    atom_link.push_attribute(("href", link));
    atom_link.push_attribute(("rel", "self"));
    atom_link.push_attribute(("type", "application/rss+xml"));
    writer.write_event(Event::Empty(atom_link))?;

    // Determine token separator for URLs (? if no existing params, & if already has ?)
    let token_joiner = if token_suffix.is_empty() {
        ""
    } else if token_suffix.starts_with('?') {
        // For URLs that already have query params, convert ? to &
        ""
    } else {
        ""
    };
    let _ = token_joiner; // used below

    // Episodes
    for post in posts {
        writer.write_event(Event::Start(BytesStart::new("item")))?;

        // Title: optionally prepend feed title for aggregate feeds
        let item_title = if prepend_feed_title {
            if let Some(ref feed_title) = post.feed_title {
                format!("[{}] {}", feed_title, post.title)
            } else {
                post.title.clone()
            }
        } else {
            post.title.clone()
        };
        write_text_element(&mut writer, "title", &item_title)?;

        write_text_element(&mut writer, "guid", &post.guid)?;

        // Description with Podly post page link (with token params)
        let post_details_url = append_token(
            &format!("{}/api/posts/{}", base_url, post.guid),
            token_suffix,
        );
        let desc_text = post.description.as_deref().unwrap_or("");
        let description_with_link = format!(
            "{}\n<p><a href=\"{}\">Podly Post Page</a></p>",
            desc_text, post_details_url
        );
        write_text_element(&mut writer, "description", &description_with_link)?;

        if let Some(ref release_date) = post.release_date {
            write_text_element(&mut writer, "pubDate", release_date)?;
        }

        // Audio enclosure — use /download (with token params)
        let audio_url = if post.processed_audio_path.is_some() {
            append_token(
                &format!("{}/api/posts/{}/download", base_url, post.guid),
                token_suffix,
            )
        } else {
            post.download_url.clone()
        };

        let mut enclosure = BytesStart::new("enclosure");
        enclosure.push_attribute(("url", audio_url.as_str()));
        enclosure.push_attribute(("type", "audio/mpeg"));
        enclosure.push_attribute(("length", "0"));
        writer.write_event(Event::Empty(enclosure))?;

        if let Some(ref image_url) = post.image_url {
            let mut img = BytesStart::new("itunes:image");
            img.push_attribute(("href", image_url.as_str()));
            writer.write_event(Event::Empty(img))?;
        }

        if let Some(duration) = post.duration {
            write_text_element(&mut writer, "itunes:duration", &duration.to_string())?;
        }

        writer.write_event(Event::End(BytesEnd::new("item")))?;
    }

    writer.write_event(Event::End(BytesEnd::new("channel")))?;
    writer.write_event(Event::End(BytesEnd::new("rss")))?;

    let result = writer.into_inner().into_inner();
    Ok(String::from_utf8(result).unwrap_or_default())
}

/// Append feed token query params to a URL
fn append_token(url: &str, token_suffix: &str) -> String {
    if token_suffix.is_empty() {
        url.to_string()
    } else if url.contains('?') {
        // URL already has query params, append with &
        format!("{}&{}", url, &token_suffix[1..]) // skip the leading ?
    } else {
        format!("{}{}", url, token_suffix)
    }
}

fn write_text_element<W: std::io::Write>(
    writer: &mut Writer<W>,
    tag: &str,
    text: &str,
) -> Result<(), quick_xml::Error> {
    writer.write_event(Event::Start(BytesStart::new(tag)))?;
    writer.write_event(Event::Text(BytesText::new(text)))?;
    writer.write_event(Event::End(BytesEnd::new(tag)))?;
    Ok(())
}
