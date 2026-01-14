"""
File: src/api/routers/audio.py
Purpose: POST /api/v1/audio/transcribe endpoint per Section 8.4
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
import time

from src.api.models.responses import AudioTranscribeResponse, LatticePath
from src.api.dependencies import get_g2p_engine, get_file_service
from src.core.g2p.byt5_engine import ByT5Engine
from src.core.g2p.ipa_utils import IPAUtils
from src.services.file_service import FileService
from src.utils.validators import ValidationError
from src.utils.logger import get_logger
from src.core.config import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Audio"])


@router.post(
    "/audio/transcribe",
    response_model=AudioTranscribeResponse,
    summary="Transcribe audio to phonemes",
    description="Transcribe an audio file to phoneme sequence with lattice output.",
    responses={
        200: {"description": "Successful transcription"},
        400: {"description": "Invalid audio file"},
        413: {"description": "File too large"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable"},
    },
)
async def transcribe_audio(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, or M4A)"),
    g2p_engine: ByT5Engine = Depends(get_g2p_engine),
    file_service: FileService = Depends(get_file_service),
) -> AudioTranscribeResponse:
    """Transcribe audio to phonemes.
    
    Accepts WAV, MP3, or M4A files up to 10MB. Validates format via
    magic bytes per REQ-SEC-001, REQ-SEC-002.
    
    Args:
        audio: Uploaded audio file.
        g2p_engine: G2P engine dependency.
        file_service: File service dependency.
        
    Returns:
        AudioTranscribeResponse with transcription and lattice.
        
    Raises:
        HTTPException: For invalid files or processing errors.
    """
    start_time = time.perf_counter()
    
    # Validate file size first
    content = await audio.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {settings.max_file_size_mb} MB limit",
        )
    
    try:
        # Save and validate file
        file_path = await file_service.save_upload(
            file_content=content,
            filename=audio.filename or "audio.wav",
        )
        
        # TODO: Implement Conformer-CTC audio processing
        # For now, we return a mock response since Conformer is not implemented
        # In production, this would:
        # 1. Preprocess audio (resample to 16kHz, generate Mel spectrogram)
        # 2. Run Conformer encoder
        # 3. Apply CTC decoding to generate phoneme lattice
        # 4. Extract top paths from lattice
        
        # Mock response for development
        # This placeholder indicates audio processing needs Conformer model
        transcription = "[Audio processing requires Conformer model]"
        ipa = "//"
        confidence = 0.0
        
        lattice_paths = [
            LatticePath(phonemes=["mock"], score=0.1),
        ]
        
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Clean up file
        await file_service.delete_after_processing(file_path)
        
        return AudioTranscribeResponse(
            transcription=transcription,
            ipa=ipa,
            confidence=confidence,
            lattice_paths=lattice_paths,
            processing_time_ms=elapsed_ms,
        )
        
    except ValidationError as e:
        # Handle magic byte mismatch
        if "Magic bytes" in e.message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio format. {e.message}",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during audio transcription",
        )
