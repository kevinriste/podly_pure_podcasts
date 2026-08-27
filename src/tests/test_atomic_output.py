"""Tests for the atomic output staging that prevents partial files from being
mistaken for completed processing."""

import pytest

from podcast_processor.podcast_processor import PodcastProcessor

# _atomic_output uses no instance state, so a bare object stands in for `self`.
_atomic_output = PodcastProcessor._atomic_output


def test_success_moves_staging_into_final(tmp_path) -> None:
    final = tmp_path / "out.mp3"

    with _atomic_output(object(), str(final)) as staging:
        assert staging != str(final)
        with open(staging, "wb") as handle:
            handle.write(b"done")

    assert final.read_bytes() == b"done"
    assert not (tmp_path / "out.mp3.partial").exists()


def test_failure_leaves_no_final_file(tmp_path) -> None:
    final = tmp_path / "out.mp3"

    with pytest.raises(RuntimeError):
        with _atomic_output(object(), str(final)) as staging:
            with open(staging, "wb") as handle:
                handle.write(b"partial")
            raise RuntimeError("chapter write blew up")

    # The whole point: a crashed build must not leave a complete-looking output,
    # otherwise a retry short-circuits and marks the job "completed".
    assert not final.exists()
    assert not (tmp_path / "out.mp3.partial").exists()


def test_stale_partial_is_cleared_before_build(tmp_path) -> None:
    final = tmp_path / "out.mp3"
    stale = tmp_path / "out.mp3.partial"
    stale.write_bytes(b"leftover from a previous crash")

    with _atomic_output(object(), str(final)) as staging:
        with open(staging, "wb") as handle:
            handle.write(b"fresh")

    assert final.read_bytes() == b"fresh"
