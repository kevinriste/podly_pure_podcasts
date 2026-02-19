from app.extensions import db
from app.models import ProcessingJob
from app.writer.actions.jobs import clear_all_jobs_action, mark_cancelled_action


def test_clear_all_jobs_action_active_only(app):
    with app.app_context():
        db.session.add_all(
            [
                ProcessingJob(
                    id="job-active-pending",
                    post_guid="guid-pending",
                    status="pending",
                    current_step=0,
                    total_steps=4,
                    progress_percentage=0.0,
                ),
                ProcessingJob(
                    id="job-active-running",
                    post_guid="guid-running",
                    status="running",
                    current_step=2,
                    total_steps=4,
                    progress_percentage=50.0,
                ),
                ProcessingJob(
                    id="job-history-completed",
                    post_guid="guid-completed",
                    status="completed",
                    current_step=4,
                    total_steps=4,
                    progress_percentage=100.0,
                ),
            ]
        )
        db.session.commit()

        removed = clear_all_jobs_action({"active_only": True})
        db.session.flush()

        assert removed == 2
        remaining_ids = {row.id for row in ProcessingJob.query.all()}
        assert remaining_ids == {"job-history-completed"}

        removed_rest = clear_all_jobs_action({})
        db.session.flush()

        assert removed_rest == 1
        assert ProcessingJob.query.count() == 0


def test_mark_cancelled_action_sets_step_name_to_reason(app):
    with app.app_context():
        job = ProcessingJob(
            id="job-cancelled",
            post_guid="guid-cancelled",
            status="pending",
            current_step=0,
            step_name="Queued for processing (priority=interactive)",
            total_steps=4,
            progress_percentage=0.0,
        )
        db.session.add(job)
        db.session.commit()

        result = mark_cancelled_action(
            {"job_id": "job-cancelled", "reason": "Cancelled by user request"}
        )
        db.session.flush()

        assert result == {"job_id": "job-cancelled", "status": "cancelled"}

        updated = db.session.get(ProcessingJob, "job-cancelled")
        assert updated is not None
        assert updated.status == "cancelled"
        assert updated.error_message == "Cancelled by user request"
        assert updated.step_name == "Cancelled by user request"
        assert updated.completed_at is not None
