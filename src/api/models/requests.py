"""
File: src/api/models/requests.py
Purpose: Pydantic request models per Section 7.2
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
from enum import Enum


class PhonemeLanguage(str, Enum):
    """Supported languages for phonemization."""
    EN_US = "en_US"
    EN_GB = "en_GB"
    FR = "fr"
    ES = "es"
    DE = "de"


class RhymeType(str, Enum):
    """Types of rhymes to search for."""
    PERFECT = "perfect"
    NEAR = "near"
    ASSONANCE = "assonance"
    CONSONANCE = "consonance"
    MULTISYLLABIC = "multisyllabic"
    ALL = "all"


class PhonemizeRequest(BaseModel):
    """Request model for text-to-IPA phonemization.
    
    Attributes:
        text: Text to convert to IPA (1-512 characters).
        language: Source language for pronunciation.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Text to convert to IPA (1-512 characters)",
        examples=["hello world", "I'm finna go to the store"],
    )
    language: PhonemeLanguage = Field(
        default=PhonemeLanguage.EN_US,
        description="Source language for pronunciation",
    )
    
    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate text is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class FindOronymsRequest(BaseModel):
    """Request model for oronym search.
    
    Attributes:
        text: Text to find oronyms for.
        ipa: IPA transcription to find oronyms for.
        language: Source language.
        max_results: Maximum number of results (1-200).
    """
    text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=512,
        description="Text to find oronyms for",
        examples=["ice cream"],
    )
    ipa: Optional[str] = Field(
        None,
        min_length=1,
        max_length=1024,
        description="IPA transcription to find oronyms for",
        examples=["/aɪskriːm/"],
    )
    language: PhonemeLanguage = Field(
        default=PhonemeLanguage.EN_US,
        description="Source language",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results to return",
    )
    
    @model_validator(mode="after")
    def at_least_one_required(self) -> "FindOronymsRequest":
        """Validate that at least one of text or ipa is provided."""
        if self.text is None and self.ipa is None:
            raise ValueError("Either 'text' or 'ipa' must be provided")
        return self


class FindRhymesRequest(BaseModel):
    """Request model for rhyme search.
    
    Attributes:
        word: Word to find rhymes for.
        rhyme_type: Type of rhymes to search for.
        min_similarity: Minimum phonetic similarity (0.50-1.00).
        max_results: Maximum number of results (1-500).
        language: Source language.
    """
    word: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Word to find rhymes for",
        examples=["nation", "abomination"],
    )
    rhyme_type: RhymeType = Field(
        default=RhymeType.ALL,
        description="Type of rhymes to search for",
    )
    min_similarity: float = Field(
        default=0.80,
        ge=0.50,
        le=1.00,
        description="Minimum phonetic similarity threshold",
    )
    max_results: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of results to return",
    )
    language: PhonemeLanguage = Field(
        default=PhonemeLanguage.EN_US,
        description="Source language",
    )
    
    @field_validator("word")
    @classmethod
    def word_not_empty(cls, v: str) -> str:
        """Validate word is not empty or whitespace only."""
        if not v.strip():
            raise ValueError("Word cannot be empty or only whitespace")
        return v.strip()


class AudioFormat(str, Enum):
    """Supported audio file formats."""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
