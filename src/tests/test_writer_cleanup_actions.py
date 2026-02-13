from app.extensions import db
from app.models import (
    Feed,
    Identification,
    ModelCall,
    Post,
    ProcessingJob,
    TranscriptSegment,
)
from app.writer.actions.cleanup import clear_post_identifications_only_action


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
