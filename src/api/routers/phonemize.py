"""
File: src/api/routers/phonemize.py
Purpose: POST /api/v1/phonemize endpoint per Section 8.1
"""

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.models.requests import PhonemizeRequest
from src.api.models.responses import PhonemizeResponse, PhonemeAlternative
from src.api.dependencies import get_g2p_engine, get_cache
from src.core.g2p.byt5_engine import ByT5Engine
from src.services.cache_service import CacheService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Phonemization"])


@router.post(
    "/phonemize",
    response_model=PhonemizeResponse,
    summary="Convert text to IPA",
    description="Convert text to International Phonetic Alphabet (IPA) representation.",
    responses={
        200: {"description": "Successful phonemization"},
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable"},
    },
)
async def phonemize(
    request: PhonemizeRequest,
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
    cache: CacheService = Depends(get_cache),
) -> PhonemizeResponse:
    """Convert text to IPA phonemes.
    
    Handles slang terms, repeated characters, numbers, and emojis per
    REQ-G2P-006 through REQ-G2P-009.
    
    Args:
        request: Phonemization request with text and language.
        g2p_engine: G2P engine dependency.
        cache: Cache service dependency.
        
    Returns:
        PhonemizeResponse with IPA and confidence.
    """
    # Check cache
    cache_key = f"{request.text}:{request.language}"
    cached = await cache.get("phoneme", cache_key)
    if cached:
        return PhonemizeResponse(**cached)
    
    try:
        # Perform G2P conversion
        result = await g2p_engine.phonemize(
            text=request.text,
            language=request.language.value,
        )
        
        # Build response
        alternatives = []
        if result.confidence < 0.9:
            for ipa, conf in result.alternatives[:5]:
                alternatives.append(PhonemeAlternative(ipa=ipa, confidence=conf))
        
        response = PhonemizeResponse(
            ipa=result.ipa,
            confidence=result.confidence,
            alternatives=alternatives,
            processing_time_ms=result.processing_time_ms,
        )
        
        # Cache result
        await cache.set("phoneme", cache_key, response.model_dump())
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Phonemization error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during phonemization",
        )
