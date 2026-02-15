from app.extensions import db
from app.models import (
    Feed,
    Identification,
    ModelCall,
    Post,
    ProcessingJob,
    TranscriptSegment,
)
from app.writer.actions.cleanup import (
    cleanup_missing_audio_paths_action,
    clear_post_identifications_only_action,
)


def test_clear_post_identifications_only_action_clears_processed_path(app):
    with app.app_context():
        feed = Feed(title="Cleanup Feed", rss_url="https://example.com/feed.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            guid="cleanup-guid",
            download_url="https://example.com/audio.mp3",
            title="Cleanup Episode",
            processed_audio_path="/tmp/processed.mp3",
            unprocessed_audio_path="/tmp/unprocessed.mp3",
            whitelisted=True,
        )
        db.session.add(post)
        db.session.commit()

        segment = TranscriptSegment(
            post_id=post.id,
            sequence_num=0,
            start_time=0.0,
            end_time=10.0,
            text="segment",
        )
        db.session.add(segment)
        db.session.commit()

        whisper_call = ModelCall(
            post_id=post.id,
            first_segment_sequence_num=0,
            last_segment_sequence_num=0,
            model_name="groq_whisper-large-v3-turbo",
            prompt="whisper",
            status="success",
        )
        llm_call = ModelCall(
            post_id=post.id,
            first_segment_sequence_num=0,
            last_segment_sequence_num=0,
            model_name="openai/gpt-5-mini",
            prompt="classify",
            status="success",
        )
        db.session.add_all([whisper_call, llm_call])
        db.session.commit()

        db.session.add(
            Identification(
                transcript_segment_id=segment.id,
                model_call_id=llm_call.id,
                label="ad",
                confidence=0.95,
            )
        )
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

        result = clear_post_identifications_only_action({"post_id": post.id})
        db.session.flush()
        db.session.refresh(post)

        assert result["post_id"] == post.id
        assert result["segments_preserved"] == 1

        assert post.processed_audio_path is None
        assert post.unprocessed_audio_path == "/tmp/unprocessed.mp3"
        assert TranscriptSegment.query.filter_by(post_id=post.id).count() == 1
        assert (
            Identification.query.join(TranscriptSegment)
            .filter(TranscriptSegment.post_id == post.id)
            .count()
            == 0
        )
        remaining_models = {
            row.model_name for row in ModelCall.query.filter_by(post_id=post.id).all()
        }
        assert remaining_models == {"groq_whisper-large-v3-turbo"}
        assert ProcessingJob.query.filter_by(post_guid=post.guid).count() == 0


def test_clear_post_identifications_only_action_preserves_whisper_prompt_calls(app):
    with app.app_context():
        feed = Feed(title="Cleanup Feed 2", rss_url="https://example.com/feed2.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            guid="cleanup-guid-2",
            download_url="https://example.com/audio2.mp3",
            title="Cleanup Episode 2",
            processed_audio_path="/tmp/processed2.mp3",
            whitelisted=True,
        )
        db.session.add(post)
        db.session.commit()

        whisper_prompt_call = ModelCall(
            post_id=post.id,
            first_segment_sequence_num=0,
            last_segment_sequence_num=0,
            model_name="custom-whisper-name",
            prompt="Whisper transcription job",
            status="success",
        )
        llm_call = ModelCall(
            post_id=post.id,
            first_segment_sequence_num=0,
            last_segment_sequence_num=0,
            model_name="openai/gpt-5-mini",
            prompt="classify",
            status="success",
        )
        db.session.add_all([whisper_prompt_call, llm_call])
        db.session.commit()

        clear_post_identifications_only_action({"post_id": post.id})
        db.session.flush()

        remaining_models = {
            row.model_name for row in ModelCall.query.filter_by(post_id=post.id).all()
        }
        assert remaining_models == {"custom-whisper-name"}


def test_cleanup_missing_audio_paths_does_not_requeue_completed_jobs(app):
    with app.app_context():
        feed = Feed(title="Cleanup Feed 3", rss_url="https://example.com/feed3.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            guid="cleanup-guid-3",
            download_url="https://example.com/audio3.mp3",
            title="Cleanup Episode 3",
            processed_audio_path="/tmp/does-not-exist-3.mp3",
            whitelisted=True,
        )
        db.session.add(post)
        db.session.add(
            ProcessingJob(
                id="cleanup-job-3",
                post_guid=post.guid,
                status="completed",
                current_step=4,
                total_steps=4,
                progress_percentage=100.0,
            )
        )
        db.session.commit()

        touched = cleanup_missing_audio_paths_action({})
        db.session.flush()
        db.session.refresh(post)
        job = ProcessingJob.query.filter_by(id="cleanup-job-3").one()

        assert touched == 1
        assert post.processed_audio_path is None
        assert job.status == "completed"


def test_cleanup_missing_audio_paths_requeues_failed_jobs(app):
    with app.app_context():
        feed = Feed(title="Cleanup Feed 4", rss_url="https://example.com/feed4.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            guid="cleanup-guid-4",
            download_url="https://example.com/audio4.mp3",
            title="Cleanup Episode 4",
            processed_audio_path="/tmp/does-not-exist-4.mp3",
            whitelisted=True,
        )
        db.session.add(post)
        db.session.add(
            ProcessingJob(
                id="cleanup-job-4",
                post_guid=post.guid,
                status="failed",
                current_step=2,
                total_steps=4,
                progress_percentage=50.0,
                step_name="Old failure",
                error_message="boom",
            )
        )
        db.session.commit()

        touched = cleanup_missing_audio_paths_action({})
        db.session.flush()
        db.session.refresh(post)
        job = ProcessingJob.query.filter_by(id="cleanup-job-4").one()

        assert touched == 1
        assert post.processed_audio_path is None
        assert job.status == "pending"
        assert job.current_step == 0
        assert job.progress_percentage == 0.0
        assert job.step_name == "Not started"
        assert job.error_message is None
        assert job.started_at is None
        assert job.completed_at is None
