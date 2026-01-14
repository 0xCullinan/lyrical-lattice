"""
File: tests/integration/test_api_oronyms.py
Purpose: Integration tests for /api/v1/find_oronyms endpoint
"""

import pytest
from httpx import AsyncClient


class TestOronymsEndpoint:
    """Integration tests for oronyms endpoint."""
    
    @pytest.mark.asyncio
    async def test_find_oronyms_with_text(self, async_client: AsyncClient):
        """Test finding oronyms with text input."""
        response = await async_client.post(
            "/api/v1/find_oronyms",
            json={"text": "ice cream", "language": "en_US"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "original_input" in data
        assert "original_ipa" in data
        assert "suggestions" in data
        assert "processing_time_ms" in data
    
    @pytest.mark.asyncio
    async def test_find_oronyms_with_ipa(self, async_client: AsyncClient):
        """Test finding oronyms with IPA input."""
        response = await async_client.post(
            "/api/v1/find_oronyms",
            json={"ipa": "/aɪskriːm/", "language": "en_US"},
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_find_oronyms_neither_text_nor_ipa_returns_400(self, async_client: AsyncClient):
        """Test that missing both text and ipa returns 400."""
        response = await async_client.post(
            "/api/v1/find_oronyms",
            json={"language": "en_US"},
        )
        assert response.status_code in (400, 422)
    
    @pytest.mark.asyncio
    async def test_find_oronyms_max_results(self, async_client: AsyncClient):
        """Test max_results parameter."""
        response = await async_client.post(
            "/api/v1/find_oronyms",
            json={"text": "hello", "max_results": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) <= 10
    
    @pytest.mark.asyncio
    async def test_find_oronyms_invalid_max_results(self, async_client: AsyncClient):
        """Test that invalid max_results returns 400."""
        response = await async_client.post(
            "/api/v1/find_oronyms",
            json={"text": "hello", "max_results": -5},
        )
        assert response.status_code in (400, 422)
