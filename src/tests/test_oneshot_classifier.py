from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models import ModelCall, Post, TranscriptSegment
from podcast_processor.oneshot_classifier import OneShotAdClassifier
from podcast_processor.oneshot_output import OneShotAdSegment
from shared.test_utils import create_standard_test_config


def _make_segment(
    segment_id: int, sequence_num: int, start: float, end: float, text: str = "seg"
) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        post_id=1,
        sequence_num=sequence_num,
        start_time=start,
        end_time=end,
        text=text,
    )


def test_parse_response_json_fallback_skips_invalid_segments() -> None:
    config = create_standard_test_config()
    classifier = OneShotAdClassifier(config=config)
    model_call = ModelCall(
        id=1,
        post_id=1,
        model_name="oneshot:test-model",
        prompt="prompt",
        first_segment_sequence_num=0,
        last_segment_sequence_num=1,
        status="pending",
    )

    response = (
        '{"ad_segments": ['
        '{"start_time": 10.0, "end_time": 20.0, "confidence": 0.91, "reason": "ad"},'
        '{"start_time": 21.0, "end_time": 25.0, "confidence": 1.5, "reason": "bad"}'
        "]}"
    )

    parsed = classifier._parse_response(response, model_call)
    assert len(parsed) == 1
    assert parsed[0].start_time == 10.0
    assert parsed[0].end_time == 20.0
    assert parsed[0].confidence == 0.91


def test_create_identifications_applies_confidence_and_deduplicates() -> None:
    config = create_standard_test_config()
    classifier = OneShotAdClassifier(config=config)
    model_call = ModelCall(
        id=42,
        post_id=1,
        model_name="oneshot:test-model",
        prompt="prompt",
        first_segment_sequence_num=0,
        last_segment_sequence_num=2,
        status="success",
    )

    transcript_segments = [
        _make_segment(1, 0, 0.0, 2.0),
        _make_segment(2, 1, 2.0, 4.0),
        _make_segment(3, 2, 4.0, 6.0),
    ]

    ad_segments = [
        OneShotAdSegment(start_time=0.0, end_time=4.0, confidence=0.9),
        OneShotAdSegment(start_time=1.0, end_time=5.0, confidence=0.95),
        OneShotAdSegment(start_time=10.0, end_time=12.0, confidence=0.2),
    ]

    with patch(
        "podcast_processor.oneshot_classifier.writer_client.action",
        return_value=SimpleNamespace(success=True, data={"inserted": 3}),
    ) as mock_action:
        inserted = classifier.create_identifications(
            ad_segments=ad_segments,
            transcript_segments=transcript_segments,
            model_call=model_call,
            min_confidence=0.7,
        )

    assert inserted == 3
    mock_action.assert_called_once()
    payload = mock_action.call_args.args[1]["identifications"]
    assert {row["transcript_segment_id"] for row in payload} == {1, 2, 3}
    assert all(row["model_call_id"] == 42 for row in payload)
    assert all(row["label"] == "ad" for row in payload)
    assert all(row["confidence"] == 0.95 for row in payload)


def test_create_identifications_keeps_low_confidence_as_candidates() -> None:
    config = create_standard_test_config()
    classifier = OneShotAdClassifier(config=config)
    model_call = ModelCall(
        id=7,
        post_id=1,
        model_name="oneshot:test-model",
        prompt="prompt",
        first_segment_sequence_num=0,
        last_segment_sequence_num=1,
        status="success",
    )

    transcript_segments = [
        _make_segment(1, 0, 10.0, 12.0),
        _make_segment(2, 1, 12.0, 14.0),
    ]

    ad_segments = [
        OneShotAdSegment(start_time=10.0, end_time=11.0, confidence=0.6),
        OneShotAdSegment(start_time=12.2, end_time=13.8, confidence=0.65),
    ]

    with patch(
        "podcast_processor.oneshot_classifier.writer_client.action",
        return_value=SimpleNamespace(success=True, data={"inserted": 2}),
    ) as mock_action:
        inserted = classifier.create_identifications(
            ad_segments=ad_segments,
            transcript_segments=transcript_segments,
            model_call=model_call,
            min_confidence=0.7,
        )

    assert inserted == 2
    payload = mock_action.call_args.args[1]["identifications"]
    assert [row["transcript_segment_id"] for row in payload] == [1, 2]
    assert all(row["model_call_id"] == 7 for row in payload)
    assert [row["label"] for row in payload] == ["ad_candidate", "ad_candidate"]
    assert [row["confidence"] for row in payload] == [0.6, 0.65]


def test_deduplicate_segments_merges_overlap_and_keeps_higher_confidence() -> None:
    config = create_standard_test_config()
    classifier = OneShotAdClassifier(config=config)

    merged = classifier._deduplicate_segments(
        [
            OneShotAdSegment(
                start_time=10.0,
                end_time=20.0,
                confidence=0.6,
                ad_type="house_ad",
                reason="first",
            ),
            OneShotAdSegment(
                start_time=18.0,
                end_time=25.0,
                confidence=0.9,
                ad_type="sponsor",
                reason="second",
            ),
            OneShotAdSegment(start_time=30.0, end_time=35.0, confidence=0.7),
        ]
    )

    assert len(merged) == 2
    assert merged[0].start_time == 10.0
    assert merged[0].end_time == 25.0
    assert merged[0].confidence == 0.9
    assert merged[1].start_time == 30.0
    assert merged[1].end_time == 35.0


def test_call_llm_uses_oneshot_api_key_and_structured_output_switch() -> None:
    config = create_standard_test_config()
    config.llm_max_retry_attempts = 1
    config.openai_max_tokens = 321
    classifier = OneShotAdClassifier(config=config)
    model_call = ModelCall(
        id=99,
        post_id=7,
        model_name="oneshot:test-model",
        prompt="prompt",
        first_segment_sequence_num=0,
        last_segment_sequence_num=5,
        status="pending",
    )

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content='{"ad_segments": []}'))]

    with (
        patch(
            "podcast_processor.oneshot_classifier.get_effective_oneshot_api_key",
            return_value="oneshot-key",
        ),
        patch(
            "podcast_processor.oneshot_classifier.model_supports_structured_outputs",
            return_value=False,
        ),
        patch(
            "podcast_processor.oneshot_classifier.model_uses_max_completion_tokens",
            return_value=True,
        ),
        patch(
            "podcast_processor.oneshot_classifier.writer_client.update",
            return_value=SimpleNamespace(success=True, data={}),
        ) as mock_update,
        patch(
            "podcast_processor.oneshot_classifier.litellm.completion",
            return_value=response,
        ) as mock_completion,
    ):
        content = classifier._call_llm(
            system_prompt="sys",
            user_prompt="user",
            model="openai/gpt-5-mini",
            model_call=model_call,
        )

    assert content == '{"ad_segments": []}'
    completion_kwargs = mock_completion.call_args.kwargs
    assert completion_kwargs["api_key"] == "oneshot-key"
    assert completion_kwargs["max_completion_tokens"] == 321
    assert completion_kwargs["response_format"] == {"type": "json_object"}
    pending_update_payload = mock_update.call_args_list[0].args[2]
    assert pending_update_payload["status"] == "pending"
    assert pending_update_payload["retry_attempts"] == 1
