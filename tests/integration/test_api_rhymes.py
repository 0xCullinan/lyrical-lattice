"""
File: tests/integration/test_api_rhymes.py
Purpose: Integration tests for /api/v1/find_rhymes endpoint
"""

import pytest
from httpx import AsyncClient


class TestRhymesEndpoint:
    """Integration tests for rhymes endpoint."""
    
    @pytest.mark.asyncio
    async def test_find_rhymes_valid_word(self, async_client: AsyncClient):
        """Test finding rhymes for a valid word."""
        response = await async_client.post(
            "/api/v1/find_rhymes",
            json={"word": "nation"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query_word" in data
        assert "query_ipa" in data
        assert "rhymes" in data
        assert "oronyms" in data
        assert "processing_time_ms" in data
    
    @pytest.mark.asyncio
    async def test_find_rhymes_empty_word_returns_400(self, async_client: AsyncClient):
        """Test that empty word returns 400."""
        response = await async_client.post(
            "/api/v1/find_rhymes",
            json={"word": ""},
        )
        assert response.status_code in (400, 422)
    
    @pytest.mark.asyncio
    async def test_find_rhymes_with_type_filter(self, async_client: AsyncClient):
        """Test finding rhymes with type filter."""
        response = await async_client.post(
            "/api/v1/find_rhymes",
            json={"word": "nation", "rhyme_type": "perfect"},
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_find_rhymes_min_similarity(self, async_client: AsyncClient):
        """Test min_similarity parameter."""
        response = await async_client.post(
            "/api/v1/find_rhymes",
            json={"word": "nation", "min_similarity": 0.90},
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_find_rhymes_invalid_min_similarity(self, async_client: AsyncClient):
        """Test that invalid min_similarity returns 400."""
        response = await async_client.post(
            "/api/v1/find_rhymes",
            json={"word": "nation", "min_similarity": 1.5},
        )
        assert response.status_code in (400, 422)
    
    @pytest.mark.asyncio
    async def test_find_rhymes_max_results(self, async_client: AsyncClient):
        """Test max_results parameter."""
        response = await async_client.post(
            "/api/v1/find_rhymes",
            json={"word": "nation", "max_results": 50},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["rhymes"]) <= 50
