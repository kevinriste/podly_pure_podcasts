from unittest.mock import MagicMock, patch

from app.models import Post, ProcessingJob
from podcast_processor.ad_classifier import AdClassifier
from podcast_processor.audio_processor import AudioProcessor
from podcast_processor.podcast_downloader import PodcastDownloader
from podcast_processor.podcast_processor import PodcastProcessor
from podcast_processor.processing_status_manager import ProcessingStatusManager
from podcast_processor.transcription_manager import TranscriptionManager
from shared.test_utils import create_standard_test_config


def _build_processor() -> PodcastProcessor:
    return PodcastProcessor(
        config=create_standard_test_config(),
        transcription_manager=MagicMock(spec=TranscriptionManager),
        ad_classifier=MagicMock(spec=AdClassifier),
        audio_processor=MagicMock(spec=AudioProcessor),
        status_manager=MagicMock(spec=ProcessingStatusManager),
        db_session=MagicMock(),
        downloader=MagicMock(spec=PodcastDownloader),
    )


def test_perform_processing_steps_dispatches_chapter_strategy() -> None:
    processor = _build_processor()
    post = Post(
        id=123,
        feed_id=1,
        guid="guid-chapter",
        download_url="https://example.com/chapter.mp3",
        title="Episode chapter",
        whitelisted=True,
    )
    job = ProcessingJob(
        id="job-chapter",
        post_guid=post.guid,
        status="pending",
        current_step=0,
        total_steps=4,
        progress_percentage=0.0,
    )

    with (
        patch.object(processor, "_perform_chapter_based_processing") as chapter_proc,
        patch.object(processor, "_perform_oneshot_processing") as oneshot_proc,
        patch.object(processor, "_perform_llm_based_processing") as llm_proc,
    ):
        processor._perform_processing_steps(
            post=post,
            job=job,
            processed_audio_path="/tmp/out.mp3",
            ad_detection_strategy="chapter",
            chapter_filter_strings="ad, sponsored",
        )

    chapter_proc.assert_called_once_with(
        post,
        job,
        "/tmp/out.mp3",
        None,
        "ad, sponsored",
    )
    oneshot_proc.assert_not_called()
    llm_proc.assert_not_called()


def test_perform_processing_steps_dispatches_oneshot_with_model_override() -> None:
    processor = _build_processor()
    post = Post(
        id=124,
        feed_id=1,
        guid="guid-oneshot",
        download_url="https://example.com/oneshot.mp3",
        title="Episode oneshot",
        whitelisted=True,
    )
    job = ProcessingJob(
        id="job-oneshot",
        post_guid=post.guid,
        status="pending",
        current_step=0,
        total_steps=4,
        progress_percentage=0.0,
    )

    with (
        patch.object(processor, "_perform_chapter_based_processing") as chapter_proc,
        patch.object(processor, "_perform_oneshot_processing") as oneshot_proc,
        patch.object(processor, "_perform_llm_based_processing") as llm_proc,
    ):
        processor._perform_processing_steps(
            post=post,
            job=job,
            processed_audio_path="/tmp/out.mp3",
            ad_detection_strategy="oneshot",
            oneshot_model_override="openai/gpt-5-mini",
        )

    oneshot_proc.assert_called_once_with(
        post,
        job,
        "/tmp/out.mp3",
        None,
        "openai/gpt-5-mini",
    )
    chapter_proc.assert_not_called()
    llm_proc.assert_not_called()


def test_perform_processing_steps_falls_back_to_llm_for_unknown_strategy() -> None:
    processor = _build_processor()
    post = Post(
        id=125,
        feed_id=1,
        guid="guid-llm",
        download_url="https://example.com/llm.mp3",
        title="Episode llm",
        whitelisted=True,
    )
    job = ProcessingJob(
        id="job-llm",
        post_guid=post.guid,
        status="pending",
        current_step=0,
        total_steps=4,
        progress_percentage=0.0,
    )

    with (
        patch.object(processor, "_perform_chapter_based_processing") as chapter_proc,
        patch.object(processor, "_perform_oneshot_processing") as oneshot_proc,
        patch.object(processor, "_perform_llm_based_processing") as llm_proc,
    ):
        processor._perform_processing_steps(
            post=post,
            job=job,
            processed_audio_path="/tmp/out.mp3",
            ad_detection_strategy="unexpected-value",
        )

    llm_proc.assert_called_once_with(post, job, "/tmp/out.mp3", None)
    chapter_proc.assert_not_called()
    oneshot_proc.assert_not_called()
