"""
File: src/detection/models.py
Purpose: Data models for all 21 wordplay device detection results
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class DeviceType(str, Enum):
    """Types of wordplay devices"""
    # Phonetic-only (13)
    HOMOPHONE = "homophone"
    ORONYM = "oronym"
    PERFECT_RHYME = "perfect_rhyme"
    SLANT_RHYME = "slant_rhyme"
    ASSONANCE = "assonance"
    CONSONANCE = "consonance"
    ALLITERATION = "alliteration"
    INTERNAL_RHYME = "internal_rhyme"
    MULTISYLLABIC_RHYME = "multisyllabic_rhyme"
    COMPOUND_RHYME = "compound_rhyme"
    ONOMATOPOEIA = "onomatopoeia"
    EUPHONY = "euphony"
    CACOPHONY = "cacophony"
    STACKED_RHYME = "stacked_rhyme"

    # Phonetic + Semantic (4)
    PUN = "pun"
    DOUBLE_ENTENDRE = "double_entendre"
    MALAPROPISM = "malapropism"
    MONDEGREEN = "mondegreen"

    # Music-specific (4)
    POLYRHYTHMIC_RHYME = "polyrhythmic_rhyme"
    BREATH_RHYME = "breath_rhyme"
    MELISMA_WORDPLAY = "melisma_wordplay"
    SAMPLE_FLIP = "sample_flip"


# =============================================================================
# PHONETIC-ONLY DEVICE MATCHES (13)
# =============================================================================

@dataclass
class HomophoneMatch:
    """Match for homophones - words with identical phonemes, different spellings"""
    phonemes: str
    words: list[str]
    confidence: float
    type: str = "homophone"


@dataclass
class OronymMatch:
    """Match for oronyms - phrases that sound identical with different word boundaries"""
    original_phonemes: list[str]
    segmentation: list[str]
    score: float
    confidence: float
    type: str = "oronym"


@dataclass
class RhymeMatch:
    """Match for perfect or slant rhymes"""
    query_phonemes: list[str]
    rhyme_portion: list[str]
    matches: list[str]
    rhyme_type: str  # "perfect", "slant", "multisyllabic"
    confidence: float


@dataclass
class AssonanceMatch:
    """Match for assonance - repeated vowel sounds"""
    vowel: str
    count: int
    word_indices: list[int]
    confidence: float
    type: str = "assonance"


@dataclass
class ConsonanceMatch:
    """Match for consonance - repeated consonant sounds"""
    consonant: str
    count: int
    word_indices: list[int]
    confidence: float
    type: str = "consonance"


@dataclass
class AlliterationMatch:
    """Match for alliteration - repeated initial consonant sounds"""
    consonant: str
    word_indices: list[int]
    run_length: int
    confidence: float
    type: str = "alliteration"


@dataclass
class InternalRhymeMatch:
    """Match for internal rhyme - rhyme within a line"""
    word_indices: tuple[int, int]
    rhyme_portion: list[str]
    type: str = "internal_rhyme"
    confidence: float = 1.0


@dataclass
class MultisyllabicRhymeMatch:
    """Match for multisyllabic rhyme - 2+ syllables matching"""
    syllables_matched: int
    matching_syllables: list[str]
    matches: list[str]
    confidence: float
    type: str = "multisyllabic_rhyme"


@dataclass
class CompoundRhymeMatch:
    """Match for compound/mosaic rhyme - multi-word phrases rhyming"""
    multi_word_phrase: list[list[str]]
    phrase_rhyme: list[str]
    single_word_matches: list[str]
    type: str = "compound_rhyme"
    confidence: float = 1.0


@dataclass
class OnomatopoeiaMatch:
    """Match for onomatopoeia - words imitating sounds"""
    word: str
    phonemes: list[str]
    type: str = "onomatopoeia"
    confidence: float = 1.0
    category: Optional[str] = None
    pattern: Optional[str] = None


@dataclass
class SoundTextureMatch:
    """Match for euphony/cacophony - pleasant/harsh sound combinations"""
    type: str  # "euphony", "cacophony", "neutral"
    euphony_score: float
    cacophony_score: float
    confidence: float


@dataclass
class StackedRhymeMatch:
    """Match for stacked rhymes - high density of rhymes in a bar"""
    rhyme_groups: dict[str, list[int]]
    total_rhymes: int
    largest_group_size: int
    density: float
    confidence: float
    type: str = "stacked_rhyme"


# =============================================================================
# PHONETIC + SEMANTIC DEVICE MATCHES (4)
# =============================================================================

@dataclass
class PunMatch:
    """Match for puns - homophones where both meanings fit context"""
    spoken_word: str
    homophones: list[str]
    context_fits: dict[str, float]
    pun_strength: float
    confidence: float
    type: str = "pun"


@dataclass
class DoubleEntendreMatch:
    """Match for double entendre - innocent surface with hidden meaning"""
    word: str
    surface_meaning: str
    hidden_meaning: str
    domain: str
    context_support: int
    confidence: float
    type: str = "double_entendre"


@dataclass
class MalapropismMatch:
    """Match for malapropism - wrong word that sounds like the right one"""
    spoken_word: str
    intended_word: str
    phonetic_similarity: float
    coherence_improvement: float
    confidence: float
    type: str = "malapropism"


@dataclass
class MondegreenMatch:
    """Match for mondegreen - misheard lyrics"""
    original_text: list[str]
    misheard_as: list[str]
    phonetic_match: float
    plausibility: float
    type: str = "mondegreen"


# =============================================================================
# MUSIC-SPECIFIC DEVICE MATCHES (4)
# =============================================================================

@dataclass
class PolyrhythmicMatch:
    """Match for polyrhythmic rhyme - rhymes on varying beat positions"""
    rhyme: str
    positions: list[tuple[int, float]]  # (bar_idx, beat_position)
    variance: float
    pattern: str  # "fully_varied", "mostly_varied", "partially_varied"
    confidence: float
    type: str = "polyrhythmic_rhyme"


@dataclass
class BreathRhymeMatch:
    """Match for breath rhyme - rhymes at natural pause points"""
    phonemes: list[str]
    rhyme_portion: list[str]
    beat_position: float
    breath_point: float
    type: str = "breath_rhyme"


@dataclass
class MelismaMatch:
    """Match for melisma wordplay - stretched syllables revealing hidden words"""
    original_phonemes: list[str]
    stretched_phonemes: list[str]
    revealed_words: list[str]
    type: str = "melisma_wordplay"
    confidence: float = 0.7


@dataclass
class SampleFlipMatch:
    """Match for sample flip - phonemes from sampled lyrics in new context"""
    original_song: str
    original_phonemes: list[str]
    new_phonemes: list[str]
    similarity: float
    type: str  # "sample_flip" or "partial_sample_flip"
    confidence: float


# =============================================================================
# UNIFIED RESULT
# =============================================================================

@dataclass
class WordplayResult:
    """Unified result from all detectors"""
    device_type: DeviceType
    match: Any  # One of the above match types
    confidence: float
    timestamp: Optional[float] = None  # When in audio this occurred
    word_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "device_type": self.device_type.value,
            "confidence": self.confidence,
        }
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.word_indices:
            result["word_indices"] = self.word_indices

        # Add match-specific fields
        if hasattr(self.match, '__dataclass_fields__'):
            for field_name in self.match.__dataclass_fields__:
                result[field_name] = getattr(self.match, field_name)

        return result


@dataclass
class DetectionResult:
    """Complete detection result for a bar/phrase"""
    input_text: Optional[str]
    input_phonemes: list[str]
    wordplay_matches: list[WordplayResult]
    processing_time_ms: int

    def to_dict(self) -> dict:
        return {
            "input_text": self.input_text,
            "input_phonemes": self.input_phonemes,
            "wordplay_matches": [m.to_dict() for m in self.wordplay_matches],
            "processing_time_ms": self.processing_time_ms,
        }
