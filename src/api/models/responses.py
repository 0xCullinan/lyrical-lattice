"""
File: src/api/models/responses.py
Purpose: Pydantic response models per Section 7.2
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OronymType(str, Enum):
    """Oronym match type classification."""
    PERFECT = "perfect"
    NEAR_PERFECT = "near_perfect"
    CLOSE = "close"


class HealthStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# --- Phonemize Responses ---

class PhonemeAlternative(BaseModel):
    """Alternative phonemization with confidence."""
    ipa: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PhonemizeResponse(BaseModel):
    """Response model for phonemization endpoint."""
    ipa: str = Field(..., description="Primary IPA transcription")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    alternatives: list[PhonemeAlternative] = Field(
        default_factory=list,
        description="Alternative transcriptions when confidence < 0.9",
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# --- Oronym Responses ---

class PhoneticBreakdown(BaseModel):
    """Phonetic breakdown of a word."""
    word: str = Field(..., description="The word")
    phonemes: list[str] = Field(..., description="List of IPA phonemes")
    syllables: list[str] = Field(..., description="List of syllables")
    stress_pattern: str = Field(..., description="Stress pattern (e.g., '10', '010')")


class OronymSuggestion(BaseModel):
    """A single oronym suggestion."""
    phrase: list[str] = Field(..., description="Words in the oronym phrase")
    ipa: str = Field(..., description="IPA transcription of the phrase")
    score: float = Field(..., ge=0.0, le=1.0, description="Match score")
    type: OronymType = Field(..., description="Match type (perfect/near_perfect/close)")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    phonetic_breakdown: list[PhoneticBreakdown] = Field(
        default_factory=list,
        description="Phonetic breakdown for each word",
    )
    explanation: Optional[str] = Field(None, description="Optional explanation")


class FindOronymsResponse(BaseModel):
    """Response model for oronym search endpoint."""
    original_input: str = Field(..., description="Original input text")
    original_ipa: str = Field(..., description="IPA of original input")
    suggestions: list[OronymSuggestion] = Field(
        default_factory=list,
        description="Oronym suggestions",
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# --- Rhyme Responses ---

class RhymeSuggestion(BaseModel):
    """A single rhyme suggestion."""
    word: str = Field(..., description="Rhyming word")
    ipa: str = Field(..., description="IPA transcription")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Phonetic similarity")
    rhyme_type: str = Field(..., description="Type of rhyme")
    syllables_matched: int = Field(..., description="Number of matching syllables")
    stress_pattern: str = Field(..., description="Stress pattern")


class FindRhymesResponse(BaseModel):
    """Response model for rhyme search endpoint."""
    query_word: str = Field(..., description="Query word")
    query_ipa: str = Field(..., description="IPA of query word")
    rhymes: list[RhymeSuggestion] = Field(
        default_factory=list,
        description="Rhyme suggestions",
    )
    oronyms: list[OronymSuggestion] = Field(
        default_factory=list,
        description="Related oronym suggestions",
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# --- Audio Responses ---

class LatticePath(BaseModel):
    """A path through the phoneme lattice."""
    phonemes: list[str] = Field(..., description="Phoneme sequence")
    score: float = Field(..., ge=0.0, le=1.0, description="Path probability")


class AudioTranscribeResponse(BaseModel):
    """Response model for audio transcription endpoint."""
    transcription: str = Field(..., description="Transcribed text")
    ipa: str = Field(..., description="IPA transcription")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    lattice_paths: list[LatticePath] = Field(
        default_factory=list,
        description="Top phoneme paths from lattice",
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


# --- Health Responses ---

class ServiceHealth(BaseModel):
    """Health status of a single service."""
    name: str = Field(..., description="Service name")
    status: HealthStatus = Field(..., description="Health status")
    latency_ms: Optional[int] = Field(None, description="Check latency in ms")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    overall_status: HealthStatus = Field(..., description="Overall system health")
    services: list[ServiceHealth] = Field(
        default_factory=list,
        description="Individual service health",
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class StatsResponse(BaseModel):
    """Response model for statistics endpoint."""
    corpus_size: int = Field(..., description="Number of entries in corpus")
    total_requests: int = Field(..., description="Total API requests processed")
    avg_response_time_ms: float = Field(..., description="Average response time in ms")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    uptime_seconds: int = Field(..., description="Server uptime in seconds")


# --- Error Responses ---

class ErrorDetail(BaseModel):
    """Detailed error information."""
    loc: list[str] = Field(..., description="Error location path")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str | list[ErrorDetail] = Field(..., description="Error details")
    error_code: Optional[str] = Field(None, description="Error code")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: Optional[str] = Field(None, description="ISO 8601 timestamp")
