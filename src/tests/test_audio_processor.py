import logging
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask

from app.extensions import db
from app.models import AudioSegment, Feed, Identification, ModelCall, Post, TranscriptSegment
from podcast_processor.audio_processor import AudioProcessor
from shared.config import Config
from shared.test_utils import create_standard_test_config


@pytest.fixture
def test_processor(
    test_config: Config,
    test_logger: logging.Logger,
) -> AudioProcessor:
    """Return an AudioProcessor instance with default dependencies for testing."""
    return AudioProcessor(config=test_config, logger=test_logger)


@pytest.fixture
def test_processor_with_mocks(
    test_config: Config,
    test_logger: logging.Logger,
    mock_db_session: MagicMock,
) -> AudioProcessor:
    """Return an AudioProcessor instance with mock dependencies for testing."""
    mock_identification_query = MagicMock()
    mock_transcript_segment_query = MagicMock()
    mock_model_call_query = MagicMock()

    return AudioProcessor(
        config=test_config,
        logger=test_logger,
        identification_query=mock_identification_query,
        transcript_segment_query=mock_transcript_segment_query,
        model_call_query=mock_model_call_query,
        db_session=mock_db_session,
    )


def test_get_ad_segments(app: Flask) -> None:
    """Test retrieving ad segments from the database"""
    # Create test data
    post = Post(id=1, title="Test Post")
    segment = TranscriptSegment(
        id=1,
        post_id=1,
        sequence_num=0,
        start_time=0.0,
        end_time=10.0,
        text="Test segment",
    )
    identification = Identification(
        transcript_segment_id=1, model_call_id=1, label="ad", confidence=0.9
    )

    with app.app_context():
        # Create mocks
        mock_identification_query = MagicMock()
        mock_query_chain = MagicMock()
        mock_identification_query.join.return_value = mock_query_chain
        mock_query_chain.join.return_value = mock_query_chain
        mock_query_chain.filter.return_value = mock_query_chain
        mock_query_chain.all.return_value = [identification]

        # Create processor with mocks
        test_processor = AudioProcessor(
            config=create_standard_test_config(),
            identification_query=mock_identification_query,
        )

        with patch.object(identification, "transcript_segment", segment):
            segments = test_processor.get_ad_segments(post)

            assert len(segments) == 1
            assert segments[0] == (0.0, 10.0)


def test_merge_ad_segments(
    test_processor_with_mocks: AudioProcessor,
) -> None:
    """Test merging of nearby ad segments"""
    duration_ms = 30000  # 30 seconds
    ad_segments = [
        (0.0, 5.0),  # 0-5s
        (6.0, 10.0),  # 6-10s - should merge with first segment
        (20.0, 25.0),  # 20-25s - should stay separate
    ]

    merged = test_processor_with_mocks.merge_ad_segments(
        duration_ms=duration_ms,
        ad_segments=ad_segments,
        min_ad_segment_length_seconds=2.0,
        min_ad_segment_separation_seconds=2.0,
    )

    # Should merge first two segments
    assert len(merged) == 2
    assert merged[0] == (0, 10000)  # 0-10s
    assert merged[1] == (20000, 25000)  # 20-25s


def test_merge_ad_segments_with_short_segments(
    test_processor_with_mocks: AudioProcessor,
) -> None:
    """Test that segments shorter than minimum length are filtered out"""
    duration_ms = 30000
    ad_segments = [
        (0.0, 1.0),  # Too short, should be filtered
        (10.0, 15.0),  # Long enough, should stay
        (20.0, 20.5),  # Too short, should be filtered
    ]

    merged = test_processor_with_mocks.merge_ad_segments(
        duration_ms=duration_ms,
        ad_segments=ad_segments,
        min_ad_segment_length_seconds=2.0,
        min_ad_segment_separation_seconds=2.0,
    )

    assert len(merged) == 1
    assert merged[0] == (10000, 15000)


def test_merge_ad_segments_end_extension(
    test_processor_with_mocks: AudioProcessor,
) -> None:
    """Test that segments near the end are extended to the end"""
    duration_ms = 30000
    ad_segments = [
        (28.0, 29.0),  # Near end, should extend to 30s
    ]

    merged = test_processor_with_mocks.merge_ad_segments(
        duration_ms=duration_ms,
        ad_segments=ad_segments,
        min_ad_segment_length_seconds=2.0,
        min_ad_segment_separation_seconds=2.0,
    )

    assert len(merged) == 1
    assert merged[0] == (28000, 30000)  # Extended to end


def test_fill_ina_speech_gaps_bridges_gap(app: Flask) -> None:
    """Gap with INA speech >= 50% is bridged into a single ad segment."""
    with app.app_context():
        feed = Feed(title="Test Feed", rss_url="http://example.com/rss.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post",
            guid="ina-gap-guid",
            download_url="http://example.com/audio.mp3",
        )
        db.session.add(post)
        db.session.commit()

        # INA says the gap [10-30s] is 80% speech (16s out of 20s)
        db.session.add(AudioSegment(post_id=post.id, start_time=10.0, end_time=26.0, label="speech"))
        db.session.add(AudioSegment(post_id=post.id, start_time=26.0, end_time=30.0, label="music"))
        db.session.commit()

        processor = AudioProcessor(
            config=create_standard_test_config(),
            db_session=db.session,
        )

        result = processor._fill_ina_speech_gaps(
            post,
            [(0.0, 10.0), (30.0, 40.0)],
            min_gap=15.0,
        )

        assert result == [(0.0, 40.0)]


def test_fill_ina_speech_gaps_skips_low_speech_gap(app: Flask) -> None:
    """Gap with INA speech < 50% is left as-is."""
    with app.app_context():
        feed = Feed(title="Test Feed", rss_url="http://example.com/rss2.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post 2",
            guid="ina-gap-guid-2",
            download_url="http://example.com/audio2.mp3",
        )
        db.session.add(post)
        db.session.commit()

        # INA says only 4s of speech in a 20s gap (20%)
        db.session.add(AudioSegment(post_id=post.id, start_time=24.0, end_time=28.0, label="speech"))
        db.session.add(AudioSegment(post_id=post.id, start_time=10.0, end_time=24.0, label="noise"))
        db.session.commit()

        processor = AudioProcessor(
            config=create_standard_test_config(),
            db_session=db.session,
        )

        result = processor._fill_ina_speech_gaps(
            post,
            [(0.0, 10.0), (30.0, 40.0)],
            min_gap=15.0,
        )

        assert result == [(0.0, 10.0), (30.0, 40.0)]


def test_fill_ina_speech_gaps_skips_large_content_gap(app: Flask) -> None:
    """Gap wider than max_gap is never bridged even when INA sees speech (it's content)."""
    with app.app_context():
        feed = Feed(title="Test Feed", rss_url="http://example.com/rss4.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post 4",
            guid="ina-gap-guid-4",
            download_url="http://example.com/audio4.mp3",
        )
        db.session.add(post)
        db.session.commit()

        # INA sees 100% speech in a 200s gap — this is content, not a Whisper drop
        db.session.add(AudioSegment(post_id=post.id, start_time=10.0, end_time=210.0, label="speech"))
        db.session.commit()

        processor = AudioProcessor(
            config=create_standard_test_config(),
            db_session=db.session,
        )

        result = processor._fill_ina_speech_gaps(
            post,
            [(0.0, 10.0), (210.0, 220.0)],
            min_gap=15.0,
            max_gap=60.0,
        )

        assert result == [(0.0, 10.0), (210.0, 220.0)]


def test_fill_ina_speech_gaps_skips_gap_with_content_identification(app: Flask) -> None:
    """Gap where LLM explicitly classified a segment as non-ad is not bridged."""
    with app.app_context():
        feed = Feed(title="Test Feed", rss_url="http://example.com/rss5.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post 5",
            guid="ina-gap-guid-5",
            download_url="http://example.com/audio5.mp3",
        )
        db.session.add(post)
        db.session.commit()

        # LLM saw and classified this segment as content (not ad)
        seg = TranscriptSegment(
            post_id=post.id, sequence_num=5,
            start_time=12.0, end_time=25.0, text="This is real content.",
        )
        db.session.add(seg)
        db.session.flush()

        mc = ModelCall(
            post_id=post.id,
            first_segment_sequence_num=5, last_segment_sequence_num=5,
            model_name="test", prompt="test", status="success",
        )
        db.session.add(mc)
        db.session.flush()

        db.session.add(Identification(
            transcript_segment_id=seg.id, model_call_id=mc.id,
            label="content", confidence=0.95,
        ))
        db.session.add(AudioSegment(post_id=post.id, start_time=10.0, end_time=30.0, label="speech"))
        db.session.commit()

        processor = AudioProcessor(
            config=create_standard_test_config(),
            db_session=db.session,
        )

        result = processor._fill_ina_speech_gaps(
            post,
            [(0.0, 10.0), (30.0, 40.0)],
            min_gap=15.0,
        )

        assert result == [(0.0, 10.0), (30.0, 40.0)]


def test_fill_ina_speech_gaps_bridges_gap_with_unidentified_transcript(app: Flask) -> None:
    """Gap where transcript exists but LLM never classified it is still bridged (chunk boundary miss)."""
    with app.app_context():
        feed = Feed(title="Test Feed", rss_url="http://example.com/rss6.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post 6",
            guid="ina-gap-guid-6",
            download_url="http://example.com/audio6.mp3",
        )
        db.session.add(post)
        db.session.commit()

        # Transcript exists but no identification — LLM chunk boundary miss
        db.session.add(TranscriptSegment(
            post_id=post.id, sequence_num=5,
            start_time=12.0, end_time=25.0, text="Buy our product at example.com.",
        ))
        # INA sees 100% speech
        db.session.add(AudioSegment(post_id=post.id, start_time=10.0, end_time=30.0, label="speech"))
        db.session.commit()

        processor = AudioProcessor(
            config=create_standard_test_config(),
            db_session=db.session,
        )

        result = processor._fill_ina_speech_gaps(
            post,
            [(0.0, 10.0), (30.0, 40.0)],
            min_gap=15.0,
        )

        assert result == [(0.0, 40.0)]


def test_fill_ina_speech_gaps_no_ina_data(app: Flask) -> None:
    """No INA data for post — gap is left unchanged."""
    with app.app_context():
        feed = Feed(title="Test Feed", rss_url="http://example.com/rss3.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post 3",
            guid="ina-gap-guid-3",
            download_url="http://example.com/audio3.mp3",
        )
        db.session.add(post)
        db.session.commit()

        processor = AudioProcessor(
            config=create_standard_test_config(),
            db_session=db.session,
        )

        result = processor._fill_ina_speech_gaps(
            post,
            [(0.0, 10.0), (30.0, 40.0)],
            min_gap=15.0,
        )

        assert result == [(0.0, 10.0), (30.0, 40.0)]


def test_process_audio(
    app: Flask,
    test_config: Config,
    test_logger: logging.Logger,
) -> None:
    """Test the process_audio method"""
    with app.app_context():
        processor = AudioProcessor(
            config=test_config, logger=test_logger, db_session=db.session
        )

        feed = Feed(title="Test Feed", rss_url="http://example.com/rss.xml")
        db.session.add(feed)
        db.session.commit()

        post = Post(
            feed_id=feed.id,
            title="Test Post",
            guid="test-audio-guid",
            download_url="http://example.com/audio.mp3",
            unprocessed_audio_path="path/to/audio.mp3",
        )
        db.session.add(post)
        db.session.commit()

        output_path = "path/to/output.mp3"

        # Set up mocks for get_ad_segments and get_audio_duration_ms
        with (
            patch.object(processor, "get_ad_segments", return_value=[(5.0, 10.0)]),
            patch(
                "podcast_processor.audio_processor.get_audio_duration_ms",
                side_effect=[30000, 24000],
            ),
            patch(
                "podcast_processor.audio_processor.clip_segments_with_fade"
            ) as mock_clip,
        ):
            # Call the method
            removed_segments = processor.process_audio(post, output_path)

            refreshed = db.session.get(Post, post.id)
            assert refreshed is not None
            assert refreshed.duration == 24.0  # processed output duration
            assert refreshed.processed_audio_path == output_path
            # The default test config extends a final ad segment to the end when
            # it is within the minimum separation threshold of the episode end.
            assert removed_segments == [(5000, 30000)]
            mock_clip.assert_called_once()
