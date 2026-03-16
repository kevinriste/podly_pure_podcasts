import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.extensions import db
from app.models import (
    Feed,
    Identification,
    ModelCall,
    Post,
    ProcessingJob,
    TranscriptSegment,
)
from app.posts import (
    clear_post_identifications_only,
    remove_associated_files,
    snapshot_post_processing_data,
)


class TestPostsFunctions:
    """Test class for functions in the app.posts module."""

    @patch("app.posts._remove_file_if_exists")
    @patch("app.posts._dedupe_and_find_existing")
    @patch("app.posts._collect_processed_paths")
    @patch("app.posts.get_and_make_download_path")
    @patch("app.posts.logger")
    def test_remove_associated_files_files_dont_exist(
        self,
        mock_logger,
        mock_get_download_path,
        mock_collect_paths,
        mock_dedupe,
        mock_remove_file,
        app,
    ):
        """Test remove_associated_files when files don't exist."""
        with app.app_context():
            # Set up mocks
            mock_collect_paths.return_value = [Path("/path/to/processed.mp3")]
            mock_dedupe.return_value = (
                [Path("/path/to/processed.mp3")],
                None,  # No existing file found
            )
            mock_get_download_path.return_value = "/path/to/unprocessed.mp3"

            # Create test post
            post = Post(id=1, title="Test Post")

            # Call the function
            remove_associated_files(post)

            # Verify _remove_file_if_exists was called for unprocessed path
            assert mock_remove_file.call_count >= 1

            # Verify debug logging for no processed file
            mock_logger.debug.assert_called()

    @patch("app.posts.writer_client")
    def test_clear_post_identifications_only_archives_processed_audio(
        self, mock_writer_client, app, tmp_path
    ):
        """Reprocess clear should archive processed audio and trigger writer cleanup."""
        with app.app_context():
            processed_audio = tmp_path / "episode.mp3"
            processed_audio.write_bytes(b"test-audio")
            post = Post(
                id=123, title="Test Episode", processed_audio_path=str(processed_audio)
            )

            mock_writer_client.action.return_value = SimpleNamespace(
                success=True, data={"segments_preserved": 12}
            )

            clear_post_identifications_only(post)

            backups = sorted(tmp_path.glob("episode.mp3.reprocess-*.bak"))
            assert len(backups) == 1
            assert backups[0].read_bytes() == b"test-audio"
            assert not processed_audio.exists()

            mock_writer_client.action.assert_called_once_with(
                "clear_post_identifications_only",
                {"post_id": 123},
                wait=True,
            )

    def test_snapshot_post_processing_data_exports_existing_state(
        self, app, tmp_path, monkeypatch
    ):
        """Snapshot should preserve transcript/model-call/identification/job state."""
        podcast_data_dir = tmp_path / "podcast-data"
        monkeypatch.setenv("PODLY_PODCAST_DATA_DIR", str(podcast_data_dir))

        with app.app_context():
            feed = Feed(title="Snapshot Feed", rss_url="https://example.com/feed.xml")
            db.session.add(feed)
            db.session.commit()

            processed_audio = tmp_path / "processed.mp3"
            processed_audio.write_bytes(b"processed")
            unprocessed_audio = tmp_path / "unprocessed.mp3"
            unprocessed_audio.write_bytes(b"unprocessed")

            post = Post(
                feed_id=feed.id,
                guid="snapshot-guid",
                download_url="https://example.com/audio.mp3",
                title="Snapshot Episode",
                processed_audio_path=str(processed_audio),
                unprocessed_audio_path=str(unprocessed_audio),
                chapter_data='{"chapters":[]}',
                whitelisted=True,
            )
            db.session.add(post)
            db.session.commit()

            segment = TranscriptSegment(
                post_id=post.id,
                sequence_num=0,
                start_time=0.0,
                end_time=5.0,
                text="hello world",
            )
            db.session.add(segment)
            db.session.commit()

            model_call = ModelCall(
                post_id=post.id,
                first_segment_sequence_num=0,
                last_segment_sequence_num=0,
                model_name="openai/gpt-5-mini",
                prompt="prompt",
                response='{"ad_segments":[]}',
                status="success",
            )
            db.session.add(model_call)
            db.session.commit()

            identification = Identification(
                transcript_segment_id=segment.id,
                model_call_id=model_call.id,
                label="ad",
                confidence=0.9,
            )
            db.session.add(identification)
            db.session.add(
                ProcessingJob(
                    post_guid=post.guid,
                    status="completed",
                    current_step=4,
                    total_steps=4,
                    progress_percentage=100.0,
                )
            )
            db.session.commit()

            snapshot_path = snapshot_post_processing_data(
                post,
                trigger="reprocess",
                force_retranscribe=False,
                requested_by_user_id=7,
            )

            assert snapshot_path is not None
            assert snapshot_path.exists()
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

            assert payload["request"]["trigger"] == "reprocess"
            assert payload["request"]["force_retranscribe"] is False
            assert payload["request"]["requested_by_user_id"] == 7
            assert payload["post"]["guid"] == "snapshot-guid"
            assert payload["counts"]["transcript_segments"] == 1
            assert payload["counts"]["model_calls"] == 1
            assert payload["counts"]["identifications"] == 1
            assert payload["counts"]["processing_jobs"] == 1
