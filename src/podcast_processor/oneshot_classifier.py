"""One-shot LLM ad classifier for podcast transcripts.

This classifier uses a single (or few) LLM call(s) to process an entire
podcast transcript at once, leveraging large-context models like GPT-5-mini.

Unlike the chunked AdClassifier, this approach:
- Sends the full transcript (or 2-hour chunks) in one call
- Skips cue detection, neighbor expansion, and boundary refinement
- Uses structured outputs when supported by the model
- Returns precise start/end timestamps directly from the LLM
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import litellm
from pydantic import ValidationError

from app.models import ModelCall, Post, TranscriptSegment
from app.writer.client import writer_client
from podcast_processor.oneshot_output import OneShotAdSegment, OneShotResponse
from shared.config import (
    Config,
    get_effective_oneshot_api_key,
    get_effective_oneshot_model,
)
from shared.llm_utils import (
    model_supports_structured_outputs,
    model_uses_max_completion_tokens,
)

# System prompt for one-shot ad detection
ONESHOT_SYSTEM_PROMPT = """You are identifying advertisements in a podcast transcript.

Your job: find the ad segments and mark how confident you are that each segment is part of an ad rather than regular show content.

confidence = how likely this segment is part of an ad, not regular show content.

Use your judgment on what counts as an ad. Sponsor reads, "brought to you by" segments, and network/house promos are typical examples. Host self-promotion (their own books, Patreon, courses) and genuine conversational product mentions are generally not ads, but use your best judgment on edge cases.

TRANSITION HANDLING (important):
When confidence shifts within an ad — especially at the entrances and exits — split those portions into separate segments with appropriately lower confidence rather than forcing one uniform block.

Examples of transition moments worth splitting out at lower confidence:
- "Thanks, [sponsor]" or sign-off lines after a sponsor read
- Repeated CTAs at the tail end of a read ("that's example.com/offer, example.com/offer")
- Opening lines that set up an ad before naming the sponsor
- Hand-backs to the show ("anyway, back to...")

When in doubt at a boundary, err toward an appropriate confidence ad segment rather than classifying it as not-an-ad-at-all. These lines are still part of the ad — just with less certainty. Content only begins when the show has genuinely resumed its normal topic.

  Boundary confidence guidance:
  - If uncertain at a boundary, prefer a one or more differential confidence transition segments rather than extending a high-confidence ad segment into likely content, cutting it inappropriately short.

Broadly: it's perfectly fine to provide segments that don't seem like ads, just provide them with appropriately lower confidence. The goal is to capture the full extent of the ad, including transition moments, rather than missing parts of it entirely. I will make a decision on my end about which confidence level items to keep.

  Return a JSON object with an "ad_segments" array. Each segment should have:
  - start_time: float (seconds)
  - end_time: float (seconds)
  - confidence: float (0.00-1.00)
  - ad_type: string (optional: "transition to ad", "beginning of ad", "ad content", "end of ad", "transition to content", or other similar descriptive label)
  - reason: string (optional: brief explanation)"""

ONESHOT_USER_PROMPT_TEMPLATE = """Podcast: "{title}"
Description: {description}
Duration: {duration:.1f} seconds
{position_note}

TRANSCRIPT:
{transcript}

Find all ad segments and a few adjacent transition segments on each side of the ad, if possible. 
Err on the side of returning multiple segments with different confidence levels, especially where your confidence in something being an ad shifts.
Strongly prefer transition-aware segmentation with confidence gradients near ad boundaries (separate lower-confidence edge segments where appropriate).
I would rather have too much to work with, lots of low-confidence segments, than have no information at all about segments in and around an ad.
Return JSON."""


class OneShotClassifyException(Exception):
    """Exception raised during one-shot classification."""


class OneShotAdClassifier:
    """Single-call LLM classifier for ad detection in podcast transcripts."""

    def __init__(
        self,
        config: Config,
        logger: Optional[logging.Logger] = None,
        db_session: Optional[Any] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger("global_logger")
        self.db_session = db_session

    def classify(
        self,
        transcript_segments: List[TranscriptSegment],
        post: Post,
        model_override: Optional[str] = None,
    ) -> List[OneShotAdSegment]:
        """
        Classify ads in transcript using one-shot LLM approach.

        Args:
            transcript_segments: List of transcript segments to classify
            post: Post containing the podcast to classify
            model_override: Optional model to use instead of config default

        Returns:
            List of detected ad segments with timestamps and confidence
        """
        if not transcript_segments:
            self.logger.info(f"No transcript segments for post {post.id}. Skipping.")
            return []

        model = model_override or get_effective_oneshot_model(self.config)
        self.logger.info(
            f"Starting one-shot classification for post {post.id} with "
            f"{len(transcript_segments)} segments using model {model}"
        )

        # Split into chunks if needed (for very long episodes)
        chunks = self._maybe_chunk_transcript(transcript_segments)
        self.logger.info(f"Processing {len(chunks)} chunk(s) for post {post.id}")

        all_segments: List[OneShotAdSegment] = []
        total_duration = transcript_segments[-1].end_time

        for i, chunk in enumerate(chunks):
            chunk_segments = self._process_chunk(
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
                post=post,
                model=model,
                total_duration=total_duration,
            )
            all_segments.extend(chunk_segments)

        # Deduplicate overlapping segments from chunk boundaries
        if len(chunks) > 1:
            all_segments = self._deduplicate_segments(all_segments)

        self.logger.info(
            f"One-shot classification complete for post {post.id}: "
            f"found {len(all_segments)} ad segments"
        )

        return all_segments

    def _maybe_chunk_transcript(
        self,
        segments: List[TranscriptSegment],
    ) -> List[List[TranscriptSegment]]:
        """
        Split transcript into chunks for very long episodes.

        Uses 2-hour chunks with 15-minute overlap to ensure ads at boundaries
        are not missed.

        Args:
            segments: All transcript segments

        Returns:
            List of segment chunks (usually just one for most episodes)
        """
        if not segments:
            return [[]]

        max_duration = int(self.config.oneshot_max_chunk_duration_seconds)
        overlap = int(self.config.oneshot_chunk_overlap_seconds)
        if overlap >= max_duration:
            self.logger.warning(
                "Invalid oneshot chunk settings (overlap=%s, max_duration=%s). "
                "Adjusting overlap to %s.",
                overlap,
                max_duration,
                max_duration - 1,
            )
            overlap = max_duration - 1

        total_duration = segments[-1].end_time - segments[0].start_time

        # Single chunk for most episodes
        if total_duration <= max_duration:
            return [segments]

        # Multi-chunk with overlap for very long episodes
        chunks: List[List[TranscriptSegment]] = []
        chunk_start_time = segments[0].start_time

        while chunk_start_time < segments[-1].end_time:
            chunk_end_time = chunk_start_time + max_duration

            # Find segments in this time range
            chunk_segments = [
                s
                for s in segments
                if s.start_time >= chunk_start_time and s.start_time < chunk_end_time
            ]

            if chunk_segments:
                chunks.append(chunk_segments)

            # Advance by (chunk_size - overlap)
            chunk_start_time += max_duration - overlap

        self.logger.info(
            f"Split {total_duration:.0f}s transcript into {len(chunks)} chunks "
            f"(max {max_duration}s each, {overlap}s overlap)"
        )

        return chunks

    def _process_chunk(
        self,
        chunk: List[TranscriptSegment],
        chunk_index: int,
        total_chunks: int,
        post: Post,
        model: str,
        total_duration: float,
    ) -> List[OneShotAdSegment]:
        """Process a single chunk of transcript segments."""
        if not chunk:
            return []

        # Build transcript text
        transcript_text = self._build_transcript_text(chunk)

        # Build position note for context
        if total_chunks == 1:
            position_note = ""
        else:
            chunk_start = chunk[0].start_time
            chunk_end = chunk[-1].end_time
            position_note = (
                f"[This is chunk {chunk_index + 1} of {total_chunks}, "
                f"covering {chunk_start:.0f}s to {chunk_end:.0f}s of {total_duration:.0f}s total]"
            )

        # Build user prompt
        user_prompt = ONESHOT_USER_PROMPT_TEMPLATE.format(
            title=post.title or "Unknown",
            description=post.description or "No description available",
            duration=total_duration,
            position_note=position_note,
            transcript=transcript_text,
        )

        # Create model call record
        model_call = self._create_model_call(
            post=post,
            chunk=chunk,
            user_prompt=user_prompt,
            model=model,
        )

        # Call LLM
        try:
            response = self._call_llm(
                system_prompt=ONESHOT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=model,
                model_call=model_call,
            )

            # Parse response
            ad_segments = self._parse_response(response, model_call)

            # Update model call status
            self._update_model_call_success(model_call, response)

            return ad_segments

        except Exception as e:
            self.logger.error(
                f"One-shot LLM call failed for post {post.id} chunk {chunk_index}: {e}",
                exc_info=True,
            )
            self._update_model_call_failed(model_call, str(e))
            raise OneShotClassifyException(
                f"LLM call failed for chunk {chunk_index}"
            ) from e

    def _build_transcript_text(self, segments: List[TranscriptSegment]) -> str:
        """Build plain transcript text with timestamps."""
        lines = []
        for seg in segments:
            lines.append(f"[{seg.start_time:.1f}] {seg.text}")
        return "\n".join(lines)

    def _create_model_call(
        self,
        post: Post,
        chunk: List[TranscriptSegment],
        user_prompt: str,
        model: str,
    ) -> ModelCall:
        """Create a ModelCall record for tracking."""
        first_seq = chunk[0].sequence_num if chunk else 0
        last_seq = chunk[-1].sequence_num if chunk else 0

        result = writer_client.action(
            "upsert_model_call",
            {
                "post_id": post.id,
                "model_name": f"oneshot:{model}",
                "first_segment_sequence_num": first_seq,
                "last_segment_sequence_num": last_seq,
                "prompt": user_prompt,
            },
            wait=True,
        )

        if not result or not result.success:
            raise RuntimeError(getattr(result, "error", "Failed to create ModelCall"))

        model_call_id = (result.data or {}).get("model_call_id")
        if model_call_id is None:
            raise RuntimeError("Writer did not return model_call_id")

        if self.db_session:
            model_call = self.db_session.get(ModelCall, int(model_call_id))
        else:
            from app.extensions import db

            model_call = db.session.get(ModelCall, int(model_call_id))

        if not isinstance(model_call, ModelCall):
            raise RuntimeError(f"ModelCall {model_call_id} not found after creation")

        return model_call

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        model_call: ModelCall,
    ) -> str:
        """Make the LLM call with structured outputs or JSON mode fallback."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion_args: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": self.config.openai_timeout,
        }

        # One-shot calls can use a dedicated ONESHOT_API_KEY override.
        oneshot_api_key = get_effective_oneshot_api_key(self.config)
        if oneshot_api_key:
            completion_args["api_key"] = oneshot_api_key
        if self.config.openai_base_url:
            completion_args["base_url"] = self.config.openai_base_url

        # Handle max tokens parameter based on model
        if model_uses_max_completion_tokens(model):
            completion_args["max_completion_tokens"] = self.config.openai_max_tokens
        else:
            completion_args["max_tokens"] = self.config.openai_max_tokens

        # Use structured outputs if supported, otherwise JSON mode
        if model_supports_structured_outputs(model):
            self.logger.info(f"Using structured outputs for model {model}")
            completion_args["response_format"] = OneShotResponse
        else:
            self.logger.info(f"Using JSON mode fallback for model {model}")
            completion_args["response_format"] = {"type": "json_object"}

        self.logger.info(
            f"Calling LLM for ModelCall {model_call.id} "
            f"(post {model_call.post_id}, segments {model_call.first_segment_sequence_num}-{model_call.last_segment_sequence_num})"
        )

        retry_count = max(
            1, int(getattr(self.config, "llm_max_retry_attempts", 1) or 1)
        )
        original_retry_attempts = (
            0 if model_call.retry_attempts is None else int(model_call.retry_attempts)
        )
        last_error: Optional[Exception] = None

        for attempt in range(retry_count):
            current_attempt_num = attempt + 1
            retry_attempts_value = original_retry_attempts + current_attempt_num

            self.logger.info(
                "Calling one-shot LLM for ModelCall %s (attempt %s/%s)",
                model_call.id,
                current_attempt_num,
                retry_count,
            )

            pending_result = writer_client.update(
                "ModelCall",
                model_call.id,
                {"status": "pending", "retry_attempts": retry_attempts_value},
                wait=True,
            )
            if not pending_result or not pending_result.success:
                raise RuntimeError(
                    getattr(pending_result, "error", "Failed to update ModelCall")
                )
            model_call.retry_attempts = retry_attempts_value

            try:
                response = litellm.completion(**completion_args)
                content = response.choices[0].message.content
                if content is None:
                    raise OneShotClassifyException("LLM returned empty response")
                return str(content)
            except Exception as e:  # pylint: disable=broad-exception-caught
                last_error = e
                self.logger.error(
                    "One-shot LLM error for ModelCall %s (attempt %s/%s): %s",
                    model_call.id,
                    current_attempt_num,
                    retry_count,
                    e,
                )
                err_result = writer_client.update(
                    "ModelCall",
                    model_call.id,
                    {
                        "error_message": str(e),
                        "retry_attempts": retry_attempts_value,
                    },
                    wait=True,
                )
                if not err_result or not err_result.success:
                    raise RuntimeError(
                        getattr(err_result, "error", "Failed to update ModelCall")
                    ) from e

                model_call.error_message = str(e)
                model_call.retry_attempts = retry_attempts_value

                if not self._is_retryable_error(e):
                    raise

                if attempt < retry_count - 1:
                    self._wait_before_retry(e, attempt)

        if last_error:
            raise last_error

        raise OneShotClassifyException(
            f"Maximum retries ({retry_count}) exceeded for ModelCall {model_call.id}"
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return (
            "503" in error_str
            or "service unavailable" in error_str
            or "rate_limit_error" in error_str
            or "ratelimiterror" in error_str
            or "429" in error_str
            or "rate limit" in error_str
        )

    def _wait_before_retry(self, error: Exception, attempt: int) -> None:
        error_str = str(error).lower()
        if any(
            term in error_str
            for term in ["rate_limit_error", "ratelimiterror", "429", "rate limit"]
        ):
            wait_time = 60 * (2**attempt)
        else:
            wait_time = 2**attempt

        self.logger.info(
            "Waiting %ss before next one-shot retry for ModelCall error: %s",
            wait_time,
            error,
        )
        time.sleep(wait_time)

    def _parse_response(
        self,
        response: str,
        model_call: ModelCall,
    ) -> List[OneShotAdSegment]:
        """Parse LLM response into ad segments."""
        try:
            # Try parsing as OneShotResponse directly
            parsed = OneShotResponse.model_validate_json(response)
            return parsed.ad_segments
        except ValidationError:
            pass

        # Try parsing as raw JSON and extracting ad_segments
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "ad_segments" in data:
                segments = []
                for seg_data in data["ad_segments"]:
                    try:
                        seg = OneShotAdSegment.model_validate(seg_data)
                        segments.append(seg)
                    except ValidationError as e:
                        self.logger.warning(
                            f"Skipping invalid segment in response: {e}"
                        )
                return segments
        except json.JSONDecodeError:
            pass

        self.logger.error(
            f"Failed to parse LLM response for ModelCall {model_call.id}: {response[:200]}..."
        )
        raise OneShotClassifyException("Failed to parse LLM response as JSON")

    def _update_model_call_success(self, model_call: ModelCall, response: str) -> None:
        """Update model call record with success status."""
        result = writer_client.update(
            "ModelCall",
            model_call.id,
            {
                "response": response,
                "status": "success",
                "error_message": None,
            },
            wait=True,
        )
        if not result or not result.success:
            self.logger.warning(
                f"Failed to update ModelCall {model_call.id} status to success"
            )

    def _update_model_call_failed(self, model_call: ModelCall, error: str) -> None:
        """Update model call record with failure status."""
        result = writer_client.update(
            "ModelCall",
            model_call.id,
            {
                "status": "failed_permanent",
                "error_message": error,
            },
            wait=True,
        )
        if not result or not result.success:
            self.logger.warning(
                f"Failed to update ModelCall {model_call.id} status to failed"
            )

    def _deduplicate_segments(
        self,
        segments: List[OneShotAdSegment],
    ) -> List[OneShotAdSegment]:
        """
        Deduplicate overlapping ad segments from chunk boundaries.

        When chunks overlap, the same ad might be detected twice. This merges
        overlapping detections, keeping the higher confidence score.
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start_time)

        merged: List[OneShotAdSegment] = []
        for seg in sorted_segments:
            if not merged:
                merged.append(seg)
                continue

            last = merged[-1]
            # Check for overlap (segments overlap if one starts before the other ends)
            if seg.start_time < last.end_time:
                # Merge: extend end time and keep higher confidence
                merged[-1] = OneShotAdSegment(
                    start_time=last.start_time,
                    end_time=max(last.end_time, seg.end_time),
                    confidence=max(last.confidence, seg.confidence),
                    ad_type=last.ad_type or seg.ad_type,
                    reason=last.reason or seg.reason,
                )
            else:
                merged.append(seg)

        if len(merged) < len(segments):
            self.logger.info(
                f"Deduplicated {len(segments)} segments down to {len(merged)}"
            )

        return merged

    def create_identifications(
        self,
        ad_segments: List[OneShotAdSegment],
        transcript_segments: List[TranscriptSegment],
        model_call: ModelCall,
        min_confidence: float,
    ) -> int:
        """
        Create Identification records from detected ad segments.

        Maps ad segment timestamps back to transcript segments and creates
        Identification records via the writer service.

        Args:
            ad_segments: Detected ad segments from LLM
            transcript_segments: Original transcript segments
            model_call: ModelCall record for linking
            min_confidence: Minimum confidence threshold

        Returns:
            Number of identifications created
        """
        if not ad_segments:
            return 0

        to_insert: List[Dict[str, Any]] = []

        # Persist one identification per transcript segment using the strongest
        # overlapping one-shot confidence. Keep below-threshold evidence as
        # non-cutting candidates for UI/debug visibility.
        for ts in transcript_segments:
            best_confidence: Optional[float] = None

            for ad_seg in ad_segments:
                if ts.start_time < ad_seg.end_time and ts.end_time > ad_seg.start_time:
                    if (
                        best_confidence is None
                        or ad_seg.confidence > best_confidence
                    ):
                        best_confidence = ad_seg.confidence

            if best_confidence is None:
                continue

            label = "ad" if best_confidence >= min_confidence else "ad_candidate"
            to_insert.append(
                {
                    "transcript_segment_id": ts.id,
                    "model_call_id": model_call.id,
                    "label": label,
                    "confidence": best_confidence,
                }
            )

        if not to_insert:
            return 0

        result = writer_client.action(
            "insert_identifications",
            {"identifications": to_insert},
            wait=True,
        )

        if not result or not result.success:
            raise RuntimeError(
                getattr(result, "error", "Failed to insert identifications")
            )

        inserted = int((result.data or {}).get("inserted") or 0)
        ad_count = sum(1 for row in to_insert if row["label"] == "ad")
        candidate_count = sum(1 for row in to_insert if row["label"] == "ad_candidate")
        self.logger.info(
            "Created %s identifications (%s ad, %s ad_candidate) from %s ad segments",
            inserted,
            ad_count,
            candidate_count,
            len(ad_segments),
        )

        return inserted

    def get_model_call_for_post(
        self,
        post: Post,
        model: str,
    ) -> Optional[ModelCall]:
        """Get the most recent successful oneshot model call for a post."""
        if self.db_session:
            query = self.db_session.query(ModelCall)
        else:
            from app.extensions import db

            query = db.session.query(ModelCall)

        result: Optional[ModelCall] = (
            query.filter_by(
                post_id=post.id,
                model_name=f"oneshot:{model}",
                status="success",
            )
            .order_by(ModelCall.timestamp.desc())
            .first()
        )
        return result
