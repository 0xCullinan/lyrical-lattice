"""
File: src/api/routers/wordplay.py
Purpose: API endpoints for wordplay detection system
"""

import time
from typing import Optional
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_g2p_engine, get_cache
from src.core.g2p.byt5_engine import ByT5Engine
from src.services.cache_service import CacheService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Wordplay"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class DeviceCategory(str, Enum):
    """Categories of wordplay devices."""
    PHONETIC = "phonetic"
    SEMANTIC = "semantic"
    MUSIC = "music"
    ALL = "all"


class WordplayDetectRequest(BaseModel):
    """Request model for wordplay detection."""
    text: Optional[str] = Field(None, description="Text input to analyze")
    phonemes: Optional[list[list[str]]] = Field(
        None,
        description="Pre-computed phonemes per word (ARPAbet format)"
    )
    words: Optional[list[str]] = Field(
        None,
        description="Word list (required if using phonemes directly)"
    )
    timing: Optional[list[dict]] = Field(
        None,
        description="Timing info for music detection: [{phonemes, timestamp}, ...]"
    )
    categories: list[DeviceCategory] = Field(
        default=[DeviceCategory.ALL],
        description="Which device categories to detect"
    )
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum results to return"
    )
    bpm: int = Field(
        default=90,
        ge=40,
        le=200,
        description="BPM for music-specific detection"
    )


class DeviceMatch(BaseModel):
    """A single detected wordplay device."""
    device_type: str = Field(..., description="Type of device detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    word_indices: list[int] = Field(
        default_factory=list,
        description="Indices of words involved"
    )
    details: dict = Field(
        default_factory=dict,
        description="Device-specific match details"
    )


class WordplayDetectResponse(BaseModel):
    """Response model for wordplay detection."""
    input_text: Optional[str] = Field(None, description="Original input text")
    input_phonemes: list[str] = Field(..., description="Flattened phoneme sequence")
    matches: list[DeviceMatch] = Field(
        default_factory=list,
        description="Detected wordplay devices"
    )
    summary: dict = Field(
        default_factory=dict,
        description="Summary statistics"
    )
    processing_time_ms: int = Field(..., description="Processing time in ms")


class RhymeSuggestRequest(BaseModel):
    """Request model for rhyme suggestions."""
    text: Optional[str] = Field(None, description="Word to find rhymes for")
    phonemes: Optional[list[str]] = Field(
        None,
        description="Phoneme sequence (ARPAbet format)"
    )
    include_slant: bool = Field(True, description="Include slant rhymes")
    include_multisyllabic: bool = Field(True, description="Include multisyllabic rhymes")
    max_results: int = Field(default=20, ge=1, le=100)


class RhymeSuggestResponse(BaseModel):
    """Response model for rhyme suggestions."""
    query_word: Optional[str] = Field(None, description="Query word")
    query_phonemes: list[str] = Field(..., description="Query phonemes")
    perfect_rhymes: list[str] = Field(default_factory=list)
    slant_rhymes: list[str] = Field(default_factory=list)
    multisyllabic_rhymes: list[str] = Field(default_factory=list)
    processing_time_ms: int = Field(..., description="Processing time in ms")


class BarAnalysisRequest(BaseModel):
    """Request model for bar analysis."""
    text: Optional[str] = Field(None, description="Bar text to analyze")
    phonemes: Optional[list[list[str]]] = Field(
        None,
        description="Pre-computed phonemes per word"
    )
    words: Optional[list[str]] = Field(None, description="Word list")


class BarAnalysisResponse(BaseModel):
    """Response model for bar analysis."""
    word_count: int = Field(..., description="Number of words")
    phoneme_count: int = Field(..., description="Number of phonemes")
    sound_texture: dict = Field(..., description="Euphony/cacophony analysis")
    rhyme_density: float = Field(..., description="Rhyme density (0-1)")
    device_counts: dict = Field(..., description="Count of each device type")
    processing_time_ms: int = Field(..., description="Processing time in ms")


# =============================================================================
# GLOBAL DETECTOR INSTANCE
# =============================================================================

# Lazy initialization of the unified detector
_detector_instance = None


def get_unified_detector():
    """Get or create the unified detector instance."""
    global _detector_instance

    if _detector_instance is None:
        import json
        from pathlib import Path
        from src.detection.unified_detector import UnifiedWordplayDetector, DetectorConfig

        # Try to load reverse index
        reverse_index = {}
        index_paths = [
            Path("/Users/macbook/Lyricist Project/Lyrics Scraper/data/master_reverse_index.json"),
            Path("data/reverse_index.json"),
            Path("reverse_index.json"),
        ]

        for path in index_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        reverse_index = json.load(f)
                    logger.info(f"Loaded reverse index from {path}: {len(reverse_index)} entries")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load reverse index from {path}: {e}")

        if not reverse_index:
            logger.warning("No reverse index found, using empty index")

        config = DetectorConfig(
            enable_phonetic=True,
            enable_semantic=True,
            enable_music=True,
            min_confidence=0.5,
        )

        _detector_instance = UnifiedWordplayDetector(
            reverse_index=reverse_index,
            config=config,
        )

    return _detector_instance


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post(
    "/detect_wordplay",
    response_model=WordplayDetectResponse,
    summary="Detect wordplay devices in text or phonemes",
    description="Analyzes input for all 21 wordplay devices including rhymes, puns, and more.",
    responses={
        200: {"description": "Successful detection"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal error"},
    },
)
async def detect_wordplay(
    request: WordplayDetectRequest,
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
    cache: CacheService = Depends(get_cache),
) -> WordplayDetectResponse:
    """Detect wordplay devices in input text or phonemes.

    Supports:
    - 13 phonetic devices (rhymes, assonance, alliteration, etc.)
    - 4 semantic devices (puns, double entendres, etc.)
    - 4 music-specific devices (polyrhythmic rhyme, breath rhyme, etc.)
    """
    start_time = time.perf_counter()

    # Validate input
    if not request.text and not request.phonemes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'phonemes' must be provided"
        )

    try:
        detector = get_unified_detector()

        # Update config based on request
        from src.detection.unified_detector import DetectorConfig
        from src.detection.models import DeviceType

        detector.config.min_confidence = request.min_confidence
        detector.config.bpm = request.bpm

        # Set enabled categories
        if DeviceCategory.ALL not in request.categories:
            detector.config.enable_phonetic = DeviceCategory.PHONETIC in request.categories
            detector.config.enable_semantic = DeviceCategory.SEMANTIC in request.categories
            detector.config.enable_music = DeviceCategory.MUSIC in request.categories
        else:
            detector.config.enable_phonetic = True
            detector.config.enable_semantic = True
            detector.config.enable_music = True

        # Get phonemes if text provided
        if request.text:
            words = request.text.split()
            bar_phonemes = []

            for word in words:
                # Check cache first
                cache_key = f"g2p:{word}"
                cached = await cache.get("phoneme", cache_key)

                if cached:
                    bar_phonemes.append(cached)
                else:
                    result = await g2p_engine.phonemize(word, "en-us")
                    # Convert IPA to ARPAbet (simplified - would use proper converter)
                    phonemes = result.ipa.split() if result.ipa else []
                    bar_phonemes.append(phonemes)
                    await cache.set("phoneme", cache_key, phonemes)
        else:
            bar_phonemes = request.phonemes
            words = request.words or [f"word_{i}" for i in range(len(bar_phonemes))]

        # Prepare timing if provided
        bar_timing = None
        if request.timing:
            bar_timing = [
                (t.get("phonemes", []), t.get("timestamp", 0.0))
                for t in request.timing
            ]

        # Run detection
        result = detector.detect_all(
            bar_phonemes=bar_phonemes,
            bar_words=words,
            bar_timing=bar_timing,
        )

        # Convert matches to response format
        matches = []
        for m in result.wordplay_matches[:request.max_results]:
            details = {}
            if hasattr(m.match, '__dataclass_fields__'):
                for field_name in m.match.__dataclass_fields__:
                    val = getattr(m.match, field_name)
                    # Convert non-serializable types
                    if isinstance(val, (list, dict, str, int, float, bool, type(None))):
                        details[field_name] = val

            matches.append(DeviceMatch(
                device_type=m.device_type.value,
                confidence=m.confidence,
                word_indices=m.word_indices,
                details=details,
            ))

        # Build summary
        summary = {
            "total_matches": len(matches),
            "device_breakdown": {},
        }
        for m in matches:
            summary["device_breakdown"][m.device_type] = (
                summary["device_breakdown"].get(m.device_type, 0) + 1
            )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return WordplayDetectResponse(
            input_text=request.text,
            input_phonemes=result.input_phonemes,
            matches=matches,
            summary=summary,
            processing_time_ms=elapsed_ms,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Wordplay detection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during wordplay detection"
        )


@router.post(
    "/suggest_rhymes",
    response_model=RhymeSuggestResponse,
    summary="Get rhyme suggestions for a word",
    description="Returns perfect, slant, and multisyllabic rhyme suggestions.",
)
async def suggest_rhymes(
    request: RhymeSuggestRequest,
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
) -> RhymeSuggestResponse:
    """Get rhyme suggestions for a word or phoneme sequence."""
    start_time = time.perf_counter()

    if not request.text and not request.phonemes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'phonemes' must be provided"
        )

    try:
        detector = get_unified_detector()

        # Get phonemes if text provided
        if request.text:
            result = await g2p_engine.phonemize(request.text, "en-us")
            phonemes = result.ipa.split() if result.ipa else []
        else:
            phonemes = request.phonemes

        suggestions = detector.get_rhyme_suggestions(
            phonemes=phonemes,
            include_slant=request.include_slant,
            include_multisyllabic=request.include_multisyllabic,
            max_results=request.max_results,
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return RhymeSuggestResponse(
            query_word=request.text,
            query_phonemes=phonemes,
            perfect_rhymes=suggestions.get("perfect", []),
            slant_rhymes=suggestions.get("slant", []),
            multisyllabic_rhymes=suggestions.get("multisyllabic", []),
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Rhyme suggestion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during rhyme suggestion"
        )


@router.post(
    "/analyze_bar",
    response_model=BarAnalysisResponse,
    summary="Analyze a bar for wordplay metrics",
    description="Returns comprehensive analysis including sound texture and device counts.",
)
async def analyze_bar(
    request: BarAnalysisRequest,
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
) -> BarAnalysisResponse:
    """Analyze a bar for wordplay characteristics."""
    start_time = time.perf_counter()

    if not request.text and not request.phonemes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'text' or 'phonemes' must be provided"
        )

    try:
        detector = get_unified_detector()

        # Get phonemes if text provided
        if request.text:
            words = request.text.split()
            bar_phonemes = []
            for word in words:
                result = await g2p_engine.phonemize(word, "en-us")
                phonemes = result.ipa.split() if result.ipa else []
                bar_phonemes.append(phonemes)
        else:
            bar_phonemes = request.phonemes
            words = request.words

        analysis = detector.analyze_bar(bar_phonemes, words)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return BarAnalysisResponse(
            word_count=analysis["word_count"],
            phoneme_count=analysis["phoneme_count"],
            sound_texture=analysis["sound_texture"],
            rhyme_density=analysis["rhyme_density"],
            device_counts=analysis["device_count"],
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        logger.error(f"Bar analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during bar analysis"
        )


@router.get(
    "/wordplay_devices",
    summary="List all supported wordplay devices",
    description="Returns information about all 21 detectable wordplay devices.",
)
async def list_wordplay_devices() -> dict:
    """List all supported wordplay devices with descriptions."""
    return {
        "phonetic_devices": [
            {"id": "homophone", "name": "Homophone", "description": "Words with identical sounds but different spellings"},
            {"id": "oronym", "name": "Oronym", "description": "Phrases that sound identical with different word boundaries"},
            {"id": "perfect_rhyme", "name": "Perfect Rhyme", "description": "Words matching from last stressed vowel to end"},
            {"id": "slant_rhyme", "name": "Slant/Near Rhyme", "description": "Phonetically similar but not identical rhymes"},
            {"id": "assonance", "name": "Assonance", "description": "Repetition of vowel sounds"},
            {"id": "consonance", "name": "Consonance", "description": "Repetition of consonant sounds"},
            {"id": "alliteration", "name": "Alliteration", "description": "Repetition of initial consonant sounds"},
            {"id": "internal_rhyme", "name": "Internal Rhyme", "description": "Rhyme within a single line"},
            {"id": "multisyllabic_rhyme", "name": "Multisyllabic Rhyme", "description": "Rhyme matching 2+ syllables"},
            {"id": "compound_rhyme", "name": "Compound Rhyme", "description": "Multi-word phrases rhyming"},
            {"id": "onomatopoeia", "name": "Onomatopoeia", "description": "Words imitating sounds"},
            {"id": "euphony", "name": "Euphony", "description": "Pleasant sound combinations"},
            {"id": "cacophony", "name": "Cacophony", "description": "Harsh sound combinations"},
            {"id": "stacked_rhyme", "name": "Stacked Rhyme", "description": "High density of rhymes in a bar"},
        ],
        "semantic_devices": [
            {"id": "pun", "name": "Pun", "description": "Homophones where both meanings fit context"},
            {"id": "double_entendre", "name": "Double Entendre", "description": "Innocent surface with hidden meaning"},
            {"id": "malapropism", "name": "Malapropism", "description": "Wrong word that sounds like the right one"},
            {"id": "mondegreen", "name": "Mondegreen", "description": "Misheard lyrics/phrases"},
        ],
        "music_devices": [
            {"id": "polyrhythmic_rhyme", "name": "Polyrhythmic Rhyme", "description": "Rhymes on varying beat positions"},
            {"id": "breath_rhyme", "name": "Breath Rhyme", "description": "Rhymes at natural pause points"},
            {"id": "melisma_wordplay", "name": "Melisma Wordplay", "description": "Stretched syllables revealing hidden words"},
            {"id": "sample_flip", "name": "Sample Flip", "description": "Phonemes from sampled lyrics in new context"},
        ],
        "total_devices": 21,
    }
