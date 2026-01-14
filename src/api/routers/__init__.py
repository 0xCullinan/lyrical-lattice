"""
File: src/api/routers/__init__.py
Purpose: API router modules
"""

from src.api.routers.phonemize import router as phonemize_router
from src.api.routers.oronyms import router as oronyms_router
from src.api.routers.rhymes import router as rhymes_router
from src.api.routers.audio import router as audio_router
from src.api.routers.health import router as health_router

__all__ = [
    "phonemize_router",
    "oronyms_router",
    "rhymes_router",
    "audio_router",
    "health_router",
]
