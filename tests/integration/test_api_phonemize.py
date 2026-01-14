"""
File: tests/integration/test_api_phonemize.py
Purpose: Integration tests for /api/v1/phonemize endpoint
"""

import pytest
from httpx import AsyncClient


class TestPhonemizeEndpoint:
    """Integration tests for phonemize endpoint."""
    
    @pytest.mark.asyncio
    async def test_phonemize_valid_text(self, async_client: AsyncClient):
        """Test phonemizing valid text."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "hello", "language": "en_US"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "ipa" in data
        assert "confidence" in data
        assert "processing_time_ms" in data
    
    @pytest.mark.asyncio
    async def test_phonemize_empty_text_returns_400(self, async_client: AsyncClient):
        """Test that empty text returns 400."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "", "language": "en_US"},
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_phonemize_whitespace_only_returns_400(self, async_client: AsyncClient):
        """Test that whitespace-only text returns 400."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "   ", "language": "en_US"},
        )
        assert response.status_code == 400
    
    @pytest.mark.asyncio
    async def test_phonemize_invalid_language(self, async_client: AsyncClient):
        """Test that invalid language returns 400."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "hello", "language": "invalid"},
        )
        # Should fail validation
        assert response.status_code in (400, 422)
    
    @pytest.mark.asyncio
    async def test_phonemize_default_language(self, async_client: AsyncClient):
        """Test that language defaults to en_US."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "hello"},
        )
        assert response.status_code == 200


class TestPhonemizeEdgeCases:
    """Edge case tests for phonemize endpoint."""
    
    @pytest.mark.asyncio
    async def test_phonemize_single_character(self, async_client: AsyncClient):
        """Test phonemizing a single character."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "a"},
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_phonemize_with_numbers(self, async_client: AsyncClient):
        """Test phonemizing text with numbers."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "4eva"},
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_phonemize_with_punctuation(self, async_client: AsyncClient):
        """Test phonemizing text with punctuation."""
        response = await async_client.post(
            "/api/v1/phonemize",
            json={"text": "Hello, world!"},
        )
        assert response.status_code == 200
