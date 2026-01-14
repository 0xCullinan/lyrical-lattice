"""
File: tests/unit/test_syllabifier.py
Purpose: Unit tests for syllabification
"""

import pytest
from src.core.g2p.syllabifier import Syllabifier


class TestSyllabifier:
    """Tests for syllabification."""
    
    @pytest.fixture
    def syllabifier(self):
        """Create syllabifier instance."""
        return Syllabifier()
    
    def test_syllabify_monosyllable(self, syllabifier):
        """Test syllabifying a monosyllable."""
        syllables = syllabifier.syllabify("/kæt/")
        assert len(syllables) >= 1
    
    def test_syllabify_disyllable(self, syllabifier):
        """Test syllabifying a disyllable."""
        syllables = syllabifier.syllabify("/hɛloʊ/")
        assert len(syllables) >= 1
    
    def test_syllabify_empty_string(self, syllabifier):
        """Test syllabifying empty string."""
        syllables = syllabifier.syllabify("")
        assert syllables == []
    
    def test_syllabify_consonant_only(self, syllabifier):
        """Test syllabifying consonant-only string."""
        syllables = syllabifier.syllabify("/str/")
        assert len(syllables) >= 1
    
    def test_count_syllables(self, syllabifier):
        """Test counting syllables."""
        count = syllabifier.count_syllables("/hɛloʊ/")
        assert count >= 1
    
    def test_get_syllable_structure(self, syllabifier):
        """Test getting syllable structure."""
        structure = syllabifier.get_syllable_structure("kæt")
        assert "onset" in structure
        assert "nucleus" in structure
        assert "coda" in structure
    
    def test_get_rhyme(self, syllabifier):
        """Test getting syllable rhyme."""
        rhyme = syllabifier.get_rhyme("kæt")
        assert len(rhyme) > 0


class TestMaximumOnsetPrinciple:
    """Tests for Maximum Onset Principle."""
    
    @pytest.fixture
    def syllabifier(self):
        """Create syllabifier instance."""
        return Syllabifier()
    
    def test_valid_onset_single_consonant(self, syllabifier):
        """Test that single consonants are valid onsets."""
        assert "p" in syllabifier.VALID_ONSETS
        assert "t" in syllabifier.VALID_ONSETS
        assert "k" in syllabifier.VALID_ONSETS
    
    def test_valid_onset_cluster(self, syllabifier):
        """Test that valid clusters are in VALID_ONSETS."""
        assert "pl" in syllabifier.VALID_ONSETS
        assert "tr" in syllabifier.VALID_ONSETS
        assert "str" in syllabifier.VALID_ONSETS
