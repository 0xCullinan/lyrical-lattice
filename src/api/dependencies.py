"""
File: src/api/dependencies.py
Purpose: FastAPI dependency injection for services
"""

from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status

from src.core.g2p.byt5_engine import ByT5Engine
from src.core.search.embedder import PhoneticEmbedder
from src.core.search.faiss_engine import FAISSEngine
from src.core.search.scorer import Scorer
from src.core.search.beam_search import BeamSearch
from src.services.db_service import DatabaseService, db_service
from src.services.cache_service import CacheService, cache_service
from src.services.file_service import FileService, file_service


# Global service instances (initialized on startup)
_g2p_engine: ByT5Engine | None = None
_embedder: PhoneticEmbedder | None = None
_faiss_engine: FAISSEngine | None = None
_scorer: Scorer | None = None
_beam_search: BeamSearch | None = None


async def initialize_services() -> None:
    """Initialize all services on application startup."""
    global _g2p_engine, _embedder, _faiss_engine, _scorer, _beam_search
    
    # Initialize infrastructure services
    await db_service.initialize()
    await cache_service.initialize()
    await file_service.initialize()
    
    # Initialize ML components
    _g2p_engine = ByT5Engine()
    await _g2p_engine.initialize()
    
    _embedder = PhoneticEmbedder()
    await _embedder.initialize()
    
    _faiss_engine = FAISSEngine()
    await _faiss_engine.initialize()
    
    # Initialize search components
    _scorer = Scorer()
    _beam_search = BeamSearch(_embedder, _faiss_engine, _scorer)


async def shutdown_services() -> None:
    """Shutdown all services gracefully per REQ-API-013."""
    await db_service.close()
    await cache_service.close()
    await file_service.close()


def get_db() -> DatabaseService:
    """Get database service.
    
    Returns:
        DatabaseService instance.
    """
    return db_service


def get_cache() -> CacheService:
    """Get cache service.
    
    Returns:
        CacheService instance.
    """
    return cache_service


def get_file_service() -> FileService:
    """Get file service.
    
    Returns:
        FileService instance.
    """
    return file_service


def get_g2p_engine() -> ByT5Engine:
    """Get G2P engine.
    
    Returns:
        ByT5Engine instance.
        
    Raises:
        HTTPException: If engine not initialized.
    """
    if _g2p_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="G2P engine not initialized",
        )
    return _g2p_engine


def get_embedder() -> PhoneticEmbedder:
    """Get phonetic embedder.
    
    Returns:
        PhoneticEmbedder instance.
        
    Raises:
        HTTPException: If embedder not initialized.
    """
    if _embedder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedder not initialized",
        )
    return _embedder


def get_faiss_engine() -> FAISSEngine:
    """Get FAISS engine.
    
    Returns:
        FAISSEngine instance.
        
    Raises:
        HTTPException: If engine not initialized.
    """
    if _faiss_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="FAISS engine not initialized",
        )
    return _faiss_engine


def get_scorer() -> Scorer:
    """Get scorer.
    
    Returns:
        Scorer instance.
        
    Raises:
        HTTPException: If scorer not initialized.
    """
    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scorer not initialized",
        )
    return _scorer


def get_beam_search() -> BeamSearch:
    """Get beam search.
    
    Returns:
        BeamSearch instance.
        
    Raises:
        HTTPException: If beam search not initialized.
    """
    if _beam_search is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Beam search not initialized",
        )
    return _beam_search
