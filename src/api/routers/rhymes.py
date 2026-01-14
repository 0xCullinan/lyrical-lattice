"""
File: src/api/routers/rhymes.py
Purpose: POST /api/v1/find_rhymes endpoint per Section 8.3
"""

from fastapi import APIRouter, Depends, HTTPException, status
import time

from src.api.models.requests import FindRhymesRequest, RhymeType
from src.api.models.responses import (
    FindRhymesResponse,
    RhymeSuggestion,
    OronymSuggestion,
    PhoneticBreakdown,
    OronymType,
    ConfidenceLevel,
)
from src.api.dependencies import get_g2p_engine, get_beam_search, get_cache
from src.core.g2p.byt5_engine import ByT5Engine
from src.core.g2p.ipa_utils import IPAUtils
from src.core.g2p.syllabifier import Syllabifier
from src.core.search.beam_search import BeamSearch
from src.services.cache_service import CacheService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Rhymes"])


@router.post(
    "/find_rhymes",
    response_model=FindRhymesResponse,
    summary="Find rhyming words",
    description="Find words that rhyme with the input word, including perfect rhymes, near rhymes, and multisyllabic rhymes.",
    responses={
        200: {"description": "Successful search"},
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable"},
    },
)
async def find_rhymes(
    request: FindRhymesRequest,
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
    beam_search: BeamSearch = Depends(get_beam_search),
    cache: CacheService = Depends(get_cache),
) -> FindRhymesResponse:
    """Find rhyming words for the input.
    
    Identifies rhymes by matching phonemes from the last stressed
    syllable onward per REQ-RHYME-001.
    
    Args:
        request: Rhyme search request.
        g2p_engine: G2P engine dependency.
        beam_search: Beam search dependency.
        cache: Cache service dependency.
        
    Returns:
        FindRhymesResponse with rhyme suggestions.
    """
    start_time = time.perf_counter()
    
    # Check cache
    cache_key = f"{request.word}:{request.rhyme_type}:{request.min_similarity}:{request.language}"
    cached = await cache.get("rhyme", cache_key)
    if cached:
        return FindRhymesResponse(**cached)
    
    try:
        # Convert word to IPA
        g2p_result = await g2p_engine.phonemize(request.word, request.language.value)
        query_ipa = g2p_result.ipa
        
        # Find rhymes
        candidates = beam_search.find_rhymes(
            query_ipa=query_ipa,
            max_results=request.max_results,
            min_similarity=request.min_similarity,
        )
        
        # Build rhyme suggestions
        syllabifier = Syllabifier()
        rhymes = []
        
        for candidate in candidates:
            word = candidate.words[0] if candidate.words else ""
            word_ipa = candidate.ipa
            
            # Count syllables matched
            query_rhyme = IPAUtils.extract_rhyme_portion(query_ipa)
            word_rhyme = IPAUtils.extract_rhyme_portion(word_ipa)
            syllables_matched = min(
                IPAUtils.count_syllables(query_rhyme),
                IPAUtils.count_syllables(word_rhyme),
            )
            
            # Classify rhyme type
            rhyme_type = beam_search.scorer.classify_rhyme_type(
                query_ipa, word_ipa, syllables_matched
            )
            
            # Filter by requested type
            if request.rhyme_type != RhymeType.ALL:
                if request.rhyme_type.value != rhyme_type:
                    continue
            
            stress_pattern = IPAUtils.get_stress_pattern(word_ipa) or "0"
            
            rhymes.append(RhymeSuggestion(
                word=word,
                ipa=word_ipa,
                similarity=candidate.phonetic_similarity,
                rhyme_type=rhyme_type,
                syllables_matched=syllables_matched,
                stress_pattern=stress_pattern,
            ))
        
        # Also find oronyms for the word
        oronym_candidates = beam_search.find_oronyms(
            query_ipa=query_ipa,
            max_results=10,
            min_similarity=0.90,
        )
        
        oronyms = []
        for candidate in oronym_candidates:
            breakdowns = []
            for word in candidate.words:
                phonemes = IPAUtils.tokenize(candidate.ipa)
                syllables = syllabifier.syllabify(candidate.ipa)
                stress = IPAUtils.get_stress_pattern(candidate.ipa) or "0"
                
                breakdowns.append(PhoneticBreakdown(
                    word=word,
                    phonemes=phonemes,
                    syllables=syllables,
                    stress_pattern=stress,
                ))
            
            if candidate.phonetic_similarity >= 1.0:
                oronym_type = OronymType.PERFECT
            elif candidate.phonetic_similarity >= 0.95:
                oronym_type = OronymType.NEAR_PERFECT
            else:
                oronym_type = OronymType.CLOSE
            
            confidence = ConfidenceLevel.HIGH if candidate.final_score >= 0.85 else (
                ConfidenceLevel.MEDIUM if candidate.final_score >= 0.70 else ConfidenceLevel.LOW
            )
            
            oronyms.append(OronymSuggestion(
                phrase=candidate.words,
                ipa=candidate.ipa,
                score=candidate.final_score,
                type=oronym_type,
                confidence=confidence,
                phonetic_breakdown=breakdowns,
            ))
        
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        response = FindRhymesResponse(
            query_word=request.word,
            query_ipa=query_ipa,
            rhymes=rhymes[:request.max_results],
            oronyms=oronyms,
            processing_time_ms=elapsed_ms,
        )
        
        # Cache result
        await cache.set("rhyme", cache_key, response.model_dump())
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Rhyme search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during rhyme search",
        )
