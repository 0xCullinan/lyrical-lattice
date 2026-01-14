"""
File: src/api/routers/oronyms.py
Purpose: POST /api/v1/find_oronyms endpoint per Section 8.2
"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.models.requests import FindOronymsRequest
from src.api.models.responses import (
    FindOronymsResponse,
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
import time

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Oronyms"])


@router.post(
    "/find_oronyms",
    response_model=FindOronymsResponse,
    summary="Find phonetically similar phrases",
    description="Find oronyms - phrases that sound identical or nearly identical but have different meanings.",
    responses={
        200: {"description": "Successful search"},
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable"},
    },
)
async def find_oronyms(
    request: FindOronymsRequest,
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
    beam_search: BeamSearch = Depends(get_beam_search),
    cache: CacheService = Depends(get_cache),
) -> FindOronymsResponse:
    """Find oronym suggestions for input text or IPA.
    
    Returns phrases that sound similar to the input, including
    multi-word alternatives like "ice cream" / "I scream".
    
    Args:
        request: Oronym search request.
        g2p_engine: G2P engine dependency.
        beam_search: Beam search dependency.
        cache: Cache service dependency.
        
    Returns:
        FindOronymsResponse with oronym suggestions.
    """
    start_time = time.perf_counter()
    
    # Determine input type and get IPA
    if request.text:
        original_input = request.text
        
        # Check cache
        cache_key = f"{request.text}:{request.language}"
        cached = await cache.get("oronym", cache_key)
        if cached:
            return FindOronymsResponse(**cached)
        
        # Convert to IPA
        g2p_result = await g2p_engine.phonemize(request.text, request.language.value)
        original_ipa = g2p_result.ipa
    else:
        original_input = request.ipa or ""
        original_ipa = request.ipa or ""
        cache_key = original_ipa
        
        cached = await cache.get("oronym", cache_key)
        if cached:
            return FindOronymsResponse(**cached)
    
    try:
        # Find oronyms using beam search
        candidates = beam_search.find_oronyms(
            query_ipa=original_ipa,
            max_results=request.max_results,
            min_similarity=0.90,  # Per REQ-ORONYM-001
        )
        
        # Build suggestions
        syllabifier = Syllabifier()
        suggestions = []
        
        for candidate in candidates:
            # Build phonetic breakdown
            breakdowns = []
            for word in candidate.words:
                # Get IPA for word (simplified - would use lookup in production)
                word_phonemes = IPAUtils.tokenize(candidate.ipa)
                syllables = syllabifier.syllabify(candidate.ipa)
                stress = IPAUtils.get_stress_pattern(candidate.ipa)
                
                breakdowns.append(PhoneticBreakdown(
                    word=word,
                    phonemes=word_phonemes,
                    syllables=syllables,
                    stress_pattern=stress or "0",
                ))
            
            # Classify type and confidence
            if candidate.phonetic_similarity >= 1.0:
                oronym_type = OronymType.PERFECT
            elif candidate.phonetic_similarity >= 0.95:
                oronym_type = OronymType.NEAR_PERFECT
            else:
                oronym_type = OronymType.CLOSE
            
            if candidate.final_score >= 0.85:
                confidence = ConfidenceLevel.HIGH
            elif candidate.final_score >= 0.70:
                confidence = ConfidenceLevel.MEDIUM
            else:
                confidence = ConfidenceLevel.LOW
            
            suggestions.append(OronymSuggestion(
                phrase=candidate.words,
                ipa=candidate.ipa,
                score=candidate.final_score,
                type=oronym_type,
                confidence=confidence,
                phonetic_breakdown=breakdowns,
                explanation=None,
            ))
        
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        response = FindOronymsResponse(
            original_input=original_input,
            original_ipa=original_ipa,
            suggestions=suggestions,
            processing_time_ms=elapsed_ms,
        )
        
        # Cache result
        await cache.set("oronym", cache_key, response.model_dump())
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Oronym search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during oronym search",
        )
