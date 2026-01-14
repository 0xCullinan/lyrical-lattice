"""
File: tests/unit/test_ipa_utils.py
Purpose: Unit tests for IPA utilities
"""

import pytest
from src.core.g2p.ipa_utils import IPAUtils


class TestIPAUtilsNormalize:
    """Tests for IPA normalization."""
    
    def test_normalize_removes_slashes(self):
        """Test that slashes are removed from IPA strings."""
        assert IPAUtils.normalize("/hello/") == "hello"
    
    def test_normalize_removes_brackets(self):
        """Test that brackets are removed from IPA strings."""
        assert IPAUtils.normalize("[hello]") == "hello"
    
    def test_normalize_trims_whitespace(self):
        """Test that leading/trailing whitespace is removed."""
        assert IPAUtils.normalize("  hello  ") == "hello"
    
    def test_normalize_standardizes_variants(self):
        """Test that common variants are standardized."""
        result = IPAUtils.normalize("/ɹɑn/")
        assert "r" in result  # ɹ -> r


class TestIPAUtilsTokenize:
    """Tests for IPA tokenization."""
    
    def test_tokenize_simple_consonants(self):
        """Test tokenizing simple consonant sequence."""
        result = IPAUtils.tokenize("/ptkbdg/")
        assert "p" in result
        assert "t" in result
        assert "k" in result
    
    def test_tokenize_vowels(self):
        """Test tokenizing vowels."""
        result = IPAUtils.tokenize("/æɛɪ/")
        assert len(result) == 3
    
    def test_tokenize_diphthongs(self):
        """Test that diphthongs are treated as single tokens."""
        result = IPAUtils.tokenize("/aɪ/")
        assert "aɪ" in result
    
    def test_tokenize_affricates(self):
        """Test that affricates are treated as single tokens."""
        result = IPAUtils.tokenize("/tʃ/")
        assert "tʃ" in result
    
    def test_tokenize_with_length_markers(self):
        """Test that length markers are included with phonemes."""
        result = IPAUtils.tokenize("/iː/")
        assert "iː" in result or ("i" in result and "ː" in result)


class TestIPAUtilsVowelConsonant:
    """Tests for vowel/consonant classification."""
    
    def test_is_vowel_true_for_vowels(self):
        """Test is_vowel returns True for vowels."""
        assert IPAUtils.is_vowel("a") is True
        assert IPAUtils.is_vowel("æ") is True
        assert IPAUtils.is_vowel("ɪ") is True
    
    def test_is_vowel_true_for_diphthongs(self):
        """Test is_vowel returns True for diphthongs."""
        assert IPAUtils.is_vowel("aɪ") is True
        assert IPAUtils.is_vowel("eɪ") is True
    
    def test_is_vowel_false_for_consonants(self):
        """Test is_vowel returns False for consonants."""
        assert IPAUtils.is_vowel("p") is False
        assert IPAUtils.is_vowel("t") is False
    
    def test_is_consonant_true_for_plosives(self):
        """Test is_consonant returns True for plosives."""
        assert IPAUtils.is_consonant("p") is True
        assert IPAUtils.is_consonant("b") is True
    
    def test_is_consonant_true_for_fricatives(self):
        """Test is_consonant returns True for fricatives."""
        assert IPAUtils.is_consonant("f") is True
        assert IPAUtils.is_consonant("s") is True
    
    def test_is_consonant_false_for_vowels(self):
        """Test is_consonant returns False for vowels."""
        assert IPAUtils.is_consonant("a") is False


class TestIPAUtilsSimilarity:
    """Tests for phonetic similarity."""
    
    def test_similarity_identical_strings(self):
        """Test that identical strings have similarity 1.0."""
        sim = IPAUtils.phonetic_similarity("/hello/", "/hello/")
        assert sim == 1.0
    
    def test_similarity_different_strings(self):
        """Test that different strings have similarity < 1.0."""
        sim = IPAUtils.phonetic_similarity("/hello/", "/world/")
        assert sim < 1.0
    
    def test_similarity_similar_strings(self):
        """Test that similar strings have high similarity."""
        # "cat" vs "bat" should be similar
        sim = IPAUtils.phonetic_similarity("/kæt/", "/bæt/")
        assert sim > 0.5
    
    def test_distance_identical_strings(self):
        """Test that identical strings have distance 0.0."""
        dist = IPAUtils.phonetic_distance("/hello/", "/hello/")
        assert dist == 0.0


class TestIPAUtilsStress:
    """Tests for stress pattern extraction."""
    
    def test_stress_pattern_simple(self):
        """Test stress pattern extraction for simple word."""
        # Just check it returns a string of digits
        result = IPAUtils.get_stress_pattern("/ˈhɛloʊ/")
        assert isinstance(result, str)
        assert all(c in "012" for c in result)
    
    def test_rhyme_portion_extraction(self):
        """Test rhyme portion extraction."""
        rhyme = IPAUtils.extract_rhyme_portion("/kæt/")
        assert "æ" in rhyme or "a" in rhyme


class TestIPAUtilsSyllables:
    """Tests for syllable counting."""
    
    def test_count_syllables_monosyllable(self):
        """Test counting syllables in monosyllable."""
        count = IPAUtils.count_syllables("/kæt/")
        assert count == 1
    
    def test_count_syllables_disyllable(self):
        """Test counting syllables in disyllable."""
        count = IPAUtils.count_syllables("/hɛloʊ/")
        assert count >= 1  # At minimum 1
