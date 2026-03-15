use std::path::Path;
use tokio::process::Command;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("ffmpeg error: {0}")]
    Ffmpeg(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Get audio duration in milliseconds using ffprobe.
pub async fn get_audio_duration_ms(audio_path: &str) -> Result<i64, AudioError> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            audio_path,
        ])
        .output()
        .await?;

    if !output.status.success() {
        return Err(AudioError::Ffmpeg("ffprobe failed".into()));
    }

    let duration_str = String::from_utf8_lossy(&output.stdout);
    let duration_sec: f64 = duration_str
        .trim()
        .parse()
        .map_err(|_| AudioError::Ffmpeg(format!("invalid duration: {duration_str}")))?;

    Ok((duration_sec * 1000.0) as i64)
}

/// Split audio into chunks of max_bytes size.
/// Returns list of (chunk_path, offset_ms) pairs.
pub async fn split_audio(
    input_path: &Path,
    output_dir: &Path,
    max_bytes: usize,
) -> Result<Vec<(std::path::PathBuf, i64)>, AudioError> {
    let input_str = input_path.to_str().unwrap_or("");
    let total_ms = get_audio_duration_ms(input_str).await?;
    let file_size = tokio::fs::metadata(input_path).await?.len() as f64;

    let total_sec = total_ms as f64 / 1000.0;
    let bytes_per_sec = file_size / total_sec;
    let chunk_duration_sec = (max_bytes as f64 / bytes_per_sec).floor() as i64;
    let chunk_duration_ms = chunk_duration_sec * 1000;

    let mut chunks = Vec::new();
    let mut offset_ms: i64 = 0;
    let mut chunk_idx = 0;

    while offset_ms < total_ms {
        let remaining = total_ms - offset_ms;
        let duration = remaining.min(chunk_duration_ms);
        let offset_sec = offset_ms as f64 / 1000.0;
        let duration_sec = duration as f64 / 1000.0;

        let ext = input_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp3");
        let chunk_path = output_dir.join(format!("chunk_{chunk_idx}.{ext}"));

        let output = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                input_str,
                "-ss",
                &format!("{offset_sec}"),
                "-t",
                &format!("{duration_sec}"),
                "-c",
                "copy",
                chunk_path.to_str().unwrap_or(""),
            ])
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AudioError::Ffmpeg(format!("chunk split failed: {stderr}")));
        }

        chunks.push((chunk_path, offset_ms));
        offset_ms += duration;
        chunk_idx += 1;
    }

    Ok(chunks)
}

/// Remove ad segments from audio with fade-in/out transitions.
pub async fn clip_segments_with_fade(
    input_path: &str,
    output_path: &str,
    ad_segments_ms: &[(i64, i64)],
    fade_ms: i64,
    use_vbr: bool,
) -> Result<(), AudioError> {
    if ad_segments_ms.is_empty() {
        // No ads to cut — just copy
        tokio::fs::copy(input_path, output_path).await?;
        return Ok(());
    }

    // Build ffmpeg filter graph to remove ad segments with fades
    let duration_ms = get_audio_duration_ms(input_path).await?;

    // Calculate content segments (inverse of ad segments)
    let mut content_segments: Vec<(i64, i64)> = Vec::new();
    let mut pos: i64 = 0;

    for (ad_start, ad_end) in ad_segments_ms {
        if *ad_start > pos {
            content_segments.push((pos, *ad_start));
        }
        pos = *ad_end;
    }
    if pos < duration_ms {
        content_segments.push((pos, duration_ms));
    }

    if content_segments.is_empty() {
        return Err(AudioError::Ffmpeg("no content segments remaining".into()));
    }

    // Build filter complex for concatenation with fades
    let mut filter_parts = Vec::new();
    let fade_sec = fade_ms as f64 / 1000.0;

    for (i, (start, end)) in content_segments.iter().enumerate() {
        let start_sec = *start as f64 / 1000.0;
        let end_sec = *end as f64 / 1000.0;
        let duration_sec = end_sec - start_sec;

        let mut segment_filter =
            format!("[0:a]atrim=start={start_sec}:end={end_sec},asetpts=PTS-STARTPTS");

        // Add fade-in (except for first segment starting at 0)
        if *start > 0 {
            segment_filter.push_str(&format!(",afade=t=in:st=0:d={fade_sec}"));
        }

        // Add fade-out (except for last segment ending at duration)
        if *end < duration_ms {
            let fade_start = (duration_sec - fade_sec).max(0.0);
            segment_filter.push_str(&format!(",afade=t=out:st={fade_start}:d={fade_sec}"));
        }

        segment_filter.push_str(&format!("[a{i}]"));
        filter_parts.push(segment_filter);
    }

    // Concatenate all segments
    let concat_inputs: String = (0..content_segments.len())
        .map(|i| format!("[a{i}]"))
        .collect();
    filter_parts.push(format!(
        "{concat_inputs}concat=n={}:v=0:a=1[out]",
        content_segments.len()
    ));

    let filter_complex = filter_parts.join(";");

    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-y",
        "-i",
        input_path,
        "-filter_complex",
        &filter_complex,
        "-map",
        "[out]",
    ]);

    if use_vbr {
        cmd.args(["-q:a", "2"]);
    } else {
        cmd.args(["-b:a", "192k"]);
    }

    cmd.arg(output_path);

    let output = cmd.output().await?;

    if output.status.success() {
        return Ok(());
    }

    // Fallback: simple approach — extract each segment separately, then concatenate
    // (matches Python's fallback when complex filter fails)
    let stderr = String::from_utf8_lossy(&output.stderr);
    tracing::warn!("Complex ffmpeg filter failed, trying simple fallback: {}", &stderr[..stderr.len().min(200)]);

    clip_segments_simple_fallback(input_path, output_path, &content_segments, fade_ms, use_vbr).await
}

/// Simple fallback: extract each content segment as a temporary file, then concatenate.
async fn clip_segments_simple_fallback(
    input_path: &str,
    output_path: &str,
    content_segments: &[(i64, i64)],
    fade_ms: i64,
    use_vbr: bool,
) -> Result<(), AudioError> {
    let temp_dir = tempfile::tempdir().map_err(|e| AudioError::Ffmpeg(format!("tempdir: {e}")))?;
    let mut segment_files = Vec::new();
    let fade_sec = fade_ms as f64 / 1000.0;

    for (i, (start, end)) in content_segments.iter().enumerate() {
        let start_sec = *start as f64 / 1000.0;
        let duration_sec = (*end - *start) as f64 / 1000.0;
        let seg_path = temp_dir.path().join(format!("seg_{i}.mp3"));

        let mut args = vec![
            "-y".to_string(), "-i".to_string(), input_path.to_string(),
            "-ss".to_string(), format!("{start_sec}"),
            "-t".to_string(), format!("{duration_sec}"),
        ];

        // Apply fades via simple filter
        let mut filters = Vec::new();
        if *start > 0 {
            filters.push(format!("afade=t=in:st=0:d={fade_sec}"));
        }
        if i < content_segments.len() - 1 {
            let fade_start = (duration_sec - fade_sec).max(0.0);
            filters.push(format!("afade=t=out:st={fade_start}:d={fade_sec}"));
        }

        if !filters.is_empty() {
            args.push("-af".to_string());
            args.push(filters.join(","));
        }

        if use_vbr {
            args.extend(["-q:a".to_string(), "2".to_string()]);
        } else {
            args.extend(["-b:a".to_string(), "192k".to_string()]);
        }

        args.push(seg_path.to_str().unwrap_or("").to_string());

        let output = Command::new("ffmpeg").args(&args).output().await?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(AudioError::Ffmpeg(format!("segment {i} extraction failed: {}", &stderr[..stderr.len().min(200)])));
        }

        segment_files.push(seg_path);
    }

    // Create concat list file
    let list_path = temp_dir.path().join("concat.txt");
    let list_content: String = segment_files
        .iter()
        .map(|p| format!("file '{}'", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    tokio::fs::write(&list_path, &list_content).await?;

    // Concatenate
    let mut concat_args = vec![
        "-y", "-f", "concat", "-safe", "0",
        "-i", list_path.to_str().unwrap_or(""),
        "-c", "copy",
    ];
    let output_str = output_path.to_string();
    concat_args.push(&output_str);

    let output = Command::new("ffmpeg").args(&concat_args).output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(AudioError::Ffmpeg(format!("concat failed: {}", &stderr[..stderr.len().min(200)])));
    }

    Ok(())
}
