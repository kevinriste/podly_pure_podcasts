"""Pydantic schemas for one-shot LLM ad detection response."""

from pydantic import BaseModel, Field


class OneShotAdSegment(BaseModel):
    """A single ad segment with precise timestamps."""

    start_time: float = Field(description="Exact start time in seconds")
    end_time: float = Field(description="Exact end time in seconds")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    ad_type: str | None = Field(
        default=None,
        description="Type of ad: 'sponsor', 'house_ad', 'transition'",
    )
    reason: str | None = Field(
        default=None,
        description="Brief explanation for why this segment is classified as an ad",
    )


class OneShotResponse(BaseModel):
    """Complete response for one-shot ad detection."""

    ad_segments: list[OneShotAdSegment] = Field(
        default_factory=list,
        description="List of detected ad segments with timestamps and confidence",
    )
