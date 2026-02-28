from app.extensions import db
from app.models import Feed, Post, ProcessingJob
from app.writer.actions.cleanup import cleanup_missing_audio_paths_action


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
