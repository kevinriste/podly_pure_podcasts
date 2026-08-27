"""Tests for byte-level audio container detection."""

from podcast_processor.audio_format import (
    detect_audio_container,
    extension_for_container,
)


def _write(path: str, data: bytes) -> str:
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def test_detects_mp4_from_ftyp_box(tmp_path) -> None:
    # ISO base-media header: 4-byte size, then `ftyp`, then major brand.
    path = _write(str(tmp_path / "episode.mp3"), b"\x00\x00\x00\x18ftypM4A \x00\x00\x02\x00")
    assert detect_audio_container(path) == "mp4"


def test_detects_mp3_from_id3_header(tmp_path) -> None:
    path = _write(str(tmp_path / "episode.mp3"), b"ID3\x04\x00\x00\x00\x00\x00\x00rest")
    assert detect_audio_container(path) == "mp3"


def test_detects_mp3_from_mpeg_frame_sync(tmp_path) -> None:
    path = _write(str(tmp_path / "episode.mp3"), b"\xff\xfb\x90\x00padding_audio")
    assert detect_audio_container(path) == "mp3"


def test_unknown_for_unrecognized_bytes(tmp_path) -> None:
    path = _write(str(tmp_path / "episode.mp3"), b"not audio at all!")
    assert detect_audio_container(path) == "unknown"


def test_unknown_for_missing_file(tmp_path) -> None:
    assert detect_audio_container(str(tmp_path / "nope.mp3")) == "unknown"


def test_extension_mapping() -> None:
    assert extension_for_container("mp3") == ".mp3"
    assert extension_for_container("mp4") == ".m4a"
    assert extension_for_container("unknown") is None
