"""
File: tests/unit/test_validators.py
Purpose: Unit tests for input validators
"""

import pytest
from pathlib import Path
from src.utils.validators import (
    validate_ipa,
    preprocess_text,
    validate_text_length,
    ValidationError,
    SLANG_MAPPINGS,
    EMOJI_MAPPINGS,
)


class TestValidateIPA:
    """Tests for IPA validation."""
    
    def test_validate_ipa_valid_string(self):
        """Test that valid IPA passes validation."""
        assert validate_ipa("/hɛloʊ/") is True
    
    def test_validate_ipa_empty_string_raises(self):
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_ipa("")
    
    def test_validate_ipa_whitespace_only_raises(self):
        """Test that whitespace-only string raises ValidationError."""
        with pytest.raises(ValidationError):
            validate_ipa("   ")


class TestPreprocessText:
    """Tests for text preprocessing."""
    
    def test_preprocess_converts_to_lowercase(self):
        """Test that preprocessing converts to lowercase."""
        result = preprocess_text("HELLO")
        assert result == result.lower()
    
    def test_preprocess_replaces_slang(self):
        """Test that slang is replaced."""
        result = preprocess_text("finna")
        assert "fixing" in result or "finna" not in result
    
    def test_preprocess_replaces_emojis(self):
        """Test that emojis are replaced."""
        result = preprocess_text("🔥")
        assert "fire" in result
    
    def test_preprocess_handles_repeated_chars(self):
        """Test that repeated characters are reduced."""
        result = preprocess_text("heeeeey")
        # Should reduce to at most 2 repeated chars
        assert "eeee" not in result
    
    def test_preprocess_empty_raises(self):
        """Test that empty text raises ValidationError."""
        with pytest.raises(ValidationError):
            preprocess_text("")
    
    def test_preprocess_whitespace_only_raises(self):
        """Test that whitespace-only raises ValidationError."""
        with pytest.raises(ValidationError):
            preprocess_text("   \n\t  ")


class TestValidateTextLength:
    """Tests for text length validation."""
    
    def test_validate_text_length_under_limit(self):
        """Test that text under limit passes."""
        result = validate_text_length("hello", max_length=512)
        assert result == "hello"
    
    def test_validate_text_length_over_limit_raises(self):
        """Test that text over limit raises."""
        with pytest.raises(ValidationError):
            validate_text_length("x" * 1000, max_length=512)


class TestSlangMappings:
    """Tests for slang mappings."""
    
    def test_slang_mappings_not_empty(self):
        """Test that slang mappings exist."""
        assert len(SLANG_MAPPINGS) > 0
    
    def test_common_slang_exists(self):
        """Test that common slang terms are mapped."""
        assert "gonna" in SLANG_MAPPINGS
        assert "wanna" in SLANG_MAPPINGS
        assert "finna" in SLANG_MAPPINGS


class TestEmojiMappings:
    """Tests for emoji mappings."""
    
    def test_emoji_mappings_not_empty(self):
        """Test that emoji mappings exist."""
        assert len(EMOJI_MAPPINGS) > 0
    
    def test_common_emojis_exist(self):
        """Test that common emojis are mapped."""
        assert "🔥" in EMOJI_MAPPINGS
        assert "❤️" in EMOJI_MAPPINGS
