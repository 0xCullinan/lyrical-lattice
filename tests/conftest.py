"""
File: tests/conftest.py
Purpose: Pytest fixtures for testing
"""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

from src.api.main import app
from src.core.g2p.byt5_engine import ByT5Engine, G2PResult
from src.core.search.embedder import PhoneticEmbedder
from src.core.search.faiss_engine import FAISSEngine
from src.core.search.scorer import Scorer
from src.services.db_service import DatabaseService
from src.services.cache_service import CacheService
from src.services.file_service import FileService


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_g2p_engine():
    """Create a mock G2P engine."""
    engine = MagicMock(spec=ByT5Engine)
    engine.is_loaded.return_value = True
    engine.phonemize = AsyncMock(return_value=G2PResult(
        ipa="/hɛloʊ wɜrld/",
        confidence=0.95,
        alternatives=[],
        processing_time_ms=42,
    ))
    return engine


@pytest.fixture
def mock_embedder():
    """Create a mock phonetic embedder."""
    import numpy as np
    
    embedder = MagicMock(spec=PhoneticEmbedder)
    embedder.is_loaded.return_value = True
    embedder.embed.return_value = np.random.randn(64).astype(np.float32)
    embedder.get_dimension.return_value = 64
    return embedder


@pytest.fixture
def mock_faiss_engine():
    """Create a mock FAISS engine."""
    engine = MagicMock(spec=FAISSEngine)
    engine.is_loaded.return_value = True
    engine.get_size.return_value = 1000
    engine.search.return_value = [
        (0, 0.95),
        (1, 0.87),
        (2, 0.82),
    ]
    engine.get_word.side_effect = lambda i: ["hello", "ice", "cream"][i] if i < 3 else None
    engine.get_ipa.side_effect = lambda i: ["/hɛloʊ/", "/aɪs/", "/kriːm/"][i] if i < 3 else None
    engine.check_health.return_value = (True, 3)
    return engine


@pytest.fixture
def mock_cache_service():
    """Create a mock cache service."""
    cache = MagicMock(spec=CacheService)
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.check_health = AsyncMock(return_value=(True, 2))
    cache.get_stats = AsyncMock(return_value={"hits": 100, "misses": 20})
    return cache


@pytest.fixture
def mock_db_service():
    """Create a mock database service."""
    db = MagicMock(spec=DatabaseService)
    db.check_health = AsyncMock(return_value=(True, 5))
    db.get_corpus_size = AsyncMock(return_value=1000)
    return db


@pytest.fixture
def mock_file_service(tmp_path):
    """Create a mock file service."""
    service = MagicMock(spec=FileService)
    service.save_upload = AsyncMock(return_value=tmp_path / "test.wav")
    service.delete_after_processing = AsyncMock()
    return service


@pytest.fixture
def scorer():
    """Create a scorer instance."""
    return Scorer()


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing endpoints."""
    # Override dependencies for testing
    from src.api import dependencies
    
    # Store original values
    orig_g2p = dependencies._g2p_engine
    orig_embedder = dependencies._embedder
    orig_faiss = dependencies._faiss_engine
    orig_scorer = dependencies._scorer
    orig_beam = dependencies._beam_search
    
    # Create mocks
    mock_g2p = MagicMock(spec=ByT5Engine)
    mock_g2p.is_loaded.return_value = True
    mock_g2p.phonemize = AsyncMock(return_value=G2PResult(
        ipa="/hɛloʊ/",
        confidence=0.95,
        alternatives=[],
        processing_time_ms=42,
    ))
    
    import numpy as np
    mock_embedder = MagicMock(spec=PhoneticEmbedder)
    mock_embedder.embed.return_value = np.random.randn(64).astype(np.float32)
    
    mock_faiss = MagicMock(spec=FAISSEngine)
    mock_faiss.is_loaded.return_value = True
    mock_faiss.get_size.return_value = 0
    mock_faiss.search.return_value = []
    mock_faiss.check_health.return_value = (True, 3)
    
    mock_scorer = Scorer()
    
    # Set mocks
    dependencies._g2p_engine = mock_g2p
    dependencies._embedder = mock_embedder
    dependencies._faiss_engine = mock_faiss
    dependencies._scorer = mock_scorer
    dependencies._beam_search = MagicMock()
    dependencies._beam_search.find_oronyms.return_value = []
    dependencies._beam_search.find_rhymes.return_value = []
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    
    # Restore
    dependencies._g2p_engine = orig_g2p
    dependencies._embedder = orig_embedder
    dependencies._faiss_engine = orig_faiss
    dependencies._scorer = orig_scorer
    dependencies._beam_search = orig_beam


# Sample test data
@pytest.fixture
def sample_ipa_strings():
    """Sample IPA strings for testing."""
    return {
        "hello": "/hɛloʊ/",
        "world": "/wɜrld/",
        "ice cream": "/aɪs kriːm/",
        "I scream": "/aɪ skriːm/",
        "abomination": "/əˌbɑməˈneɪʃən/",
        "nation": "/ˈneɪʃən/",
    }


@pytest.fixture
def sample_phoneme_list():
    """Sample phoneme lists for testing."""
    return {
        "hello": ["h", "ɛ", "l", "oʊ"],
        "world": ["w", "ɜ", "r", "l", "d"],
        "ice cream": ["aɪ", "s", "k", "r", "iː", "m"],
    }
