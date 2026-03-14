use quick_xml::Writer;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use std::io::Cursor;

use crate::db::models::{Feed, Post};

/// Generate an RSS 2.0 feed XML string for a feed with its posts.
pub fn generate_rss_feed(
    feed: &Feed,
    posts: &[Post],
    base_url: &str,
) -> Result<String, quick_xml::Error> {
    let mut writer = Writer::new(Cursor::new(Vec::new()));

    writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    // <rss>
    let mut rss = BytesStart::new("rss");
    rss.push_attribute(("version", "2.0"));
    rss.push_attribute(("xmlns:itunes", "http://www.itunes.com/dtds/podcast-1.0.dtd"));
    rss.push_attribute(("xmlns:atom", "http://www.w3.org/2005/Atom"));
    writer.write_event(Event::Start(rss))?;

    // <channel>
    writer.write_event(Event::Start(BytesStart::new("channel")))?;

    write_text_element(&mut writer, "title", &feed.title)?;
    if let Some(ref desc) = feed.description {
        write_text_element(&mut writer, "description", desc)?;
    }
    write_text_element(&mut writer, "link", &format!("{base_url}/feed/{}", feed.id))?;

    if let Some(ref image_url) = feed.image_url {
        let mut itunes_image = BytesStart::new("itunes:image");
        itunes_image.push_attribute(("href", image_url.as_str()));
        writer.write_event(Event::Empty(itunes_image))?;
    }

    // Self-referencing atom link
    let mut atom_link = BytesStart::new("atom:link");
    atom_link.push_attribute(("href", format!("{base_url}/feed/{}", feed.id).as_str()));
    atom_link.push_attribute(("rel", "self"));
    atom_link.push_attribute(("type", "application/rss+xml"));
    writer.write_event(Event::Empty(atom_link))?;

    // Episodes
    for post in posts {
        if !post.whitelisted {
            continue;
        }

        writer.write_event(Event::Start(BytesStart::new("item")))?;

        write_text_element(&mut writer, "title", &post.title)?;
        write_text_element(&mut writer, "guid", &post.guid)?;

        if let Some(ref desc) = post.description {
            write_text_element(&mut writer, "description", desc)?;
        }
        if let Some(ref release_date) = post.release_date {
            write_text_element(&mut writer, "pubDate", release_date)?;
        }

        // Audio enclosure
        let audio_url = if post.processed_audio_path.is_some() {
            format!("{base_url}/api/posts/{}/audio", post.guid)
        } else {
            post.download_url.clone()
        };

        let mut enclosure = BytesStart::new("enclosure");
        enclosure.push_attribute(("url", audio_url.as_str()));
        enclosure.push_attribute(("type", "audio/mpeg"));
        // Length is approximate for processed audio
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
