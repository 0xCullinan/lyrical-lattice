"""
File: src/core/g2p/ipa_utils.py
Purpose: IPA validation, normalization, and conversion utilities
"""

import re
from typing import Optional

# IPA consonant categories
PLOSIVES = set("pbtdkɡʔ")
NASALS = set("mnŋɲɴ")
FRICATIVES = set("fvθðszʃʒçxɣhɦ")
AFFRICATES = {"tʃ", "dʒ", "ts", "dz"}
APPROXIMANTS = set("ɹjwlɫ")
TRILLS = set("rʀ")
TAPS = set("ɾɽ")

# IPA vowel categories
CLOSE_VOWELS = set("iyɨʉɯu")
CLOSE_MID_VOWELS = set("eøɘɵɤo")
OPEN_MID_VOWELS = set("ɛœɜɞʌɔ")
OPEN_VOWELS = set("æaɶɑɒ")
SCHWA = set("əɐ")

ALL_VOWELS = CLOSE_VOWELS | CLOSE_MID_VOWELS | OPEN_MID_VOWELS | OPEN_VOWELS | SCHWA

# Common diphthongs
DIPHTHONGS = {
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ",
    "ɪə", "eə", "ʊə", "ɪɚ", "ɛɚ", "ʊɚ",
}

# Stress and length markers
STRESS_MARKERS = set("ˈˌ")
LENGTH_MARKERS = set("ːˑ")

# Syllable boundary marker
SYLLABLE_BOUNDARY = "."


class IPAUtils:
    """Utility class for IPA string manipulation.
    
    Provides methods for validation, normalization, tokenization,
    and analysis of IPA transcriptions.
    """
    
    # Regex patterns for IPA parsing
    PHONEME_PATTERN = re.compile(
        r"(?:tʃ|dʒ|ts|dz)|"  # Affricates (multi-char)
        r"(?:[eaɔɪʊə][ɪʊəɚ])|"  # Diphthongs
        r"[" + 
        "".join(PLOSIVES | NASALS | FRICATIVES | APPROXIMANTS | 
                TRILLS | TAPS | ALL_VOWELS) +
        r"]" +
        r"[ːˑ]?"  # Optional length marker
    )
    
    @staticmethod
    def normalize(ipa_string: str) -> str:
        """Normalize an IPA string.
        
        Removes delimiters, normalizes whitespace, and standardizes
        common variant characters.
        
        Args:
            ipa_string: Raw IPA string.
            
        Returns:
            Normalized IPA string.
        """
        # Remove common delimiters
        result = ipa_string.strip().strip("/[]")
        
        # Normalize common variants
        replacements = {
            "ɹ": "r",   # Some sources use ɹ, others r
            "ɡ": "g",   # IPA g vs ASCII g
            "'": "ˈ",   # ASCII apostrophe to IPA stress
            ":": "ː",   # ASCII colon to IPA length
            "ɝ": "ɜr",  # Rhoticized schwa
            "ɚ": "ər",  # Rhoticized schwa
        }
        
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        # Normalize whitespace
        result = " ".join(result.split())
        
        return result
    
    @staticmethod
    def tokenize(ipa_string: str) -> list[str]:
        """Tokenize an IPA string into individual phonemes.
        
        Handles diphthongs and affricates as single tokens.
        
        Args:
            ipa_string: IPA string to tokenize.
            
        Returns:
            List of phoneme tokens.
        """
        normalized = IPAUtils.normalize(ipa_string)
        phonemes = []
        
        i = 0
        while i < len(normalized):
            char = normalized[i]
            
            # Skip whitespace and syllable markers
            if char in " .":
                i += 1
                continue
            
            # Skip stress markers (but keep them in output)
            if char in STRESS_MARKERS:
                i += 1
                continue
            
            # Check for multi-character phonemes
            # Affricates
            if i + 1 < len(normalized):
                digraph = normalized[i:i+2]
                if digraph in AFFRICATES or digraph in DIPHTHONGS:
                    phonemes.append(digraph)
                    i += 2
                    # Check for length marker
                    if i < len(normalized) and normalized[i] in LENGTH_MARKERS:
                        phonemes[-1] += normalized[i]
                        i += 1
                    continue
            
            # Single character phoneme
            phoneme = char
            i += 1
            
            # Add length marker if present
            if i < len(normalized) and normalized[i] in LENGTH_MARKERS:
                phoneme += normalized[i]
                i += 1
            
            phonemes.append(phoneme)
        
        return phonemes
    
    @staticmethod
    def is_vowel(phoneme: str) -> bool:
        """Check if a phoneme is a vowel.
        
        Args:
            phoneme: IPA phoneme.
            
        Returns:
            True if vowel.
        """
        # Remove length markers
        base = phoneme.rstrip("ːˑ")
        
        # Check single vowels
        if base in ALL_VOWELS:
            return True
        
        # Check diphthongs
        if base in DIPHTHONGS:
            return True
        
        return False
    
    @staticmethod
    def is_consonant(phoneme: str) -> bool:
        """Check if a phoneme is a consonant.
        
        Args:
            phoneme: IPA phoneme.
            
        Returns:
            True if consonant.
        """
        base = phoneme.rstrip("ːˑ")
        
        if base in AFFRICATES:
            return True
        
        return base in (PLOSIVES | NASALS | FRICATIVES | 
                       APPROXIMANTS | TRILLS | TAPS)
    
    @staticmethod
    def get_stress_pattern(ipa_string: str) -> str:
        """Extract stress pattern from IPA string.
        
        Args:
            ipa_string: IPA transcription with stress markers.
            
        Returns:
            Stress pattern string (e.g., "10", "010", "0120").
            1 = primary stress, 2 = secondary stress, 0 = unstressed.
        """
        normalized = IPAUtils.normalize(ipa_string)
        phonemes = IPAUtils.tokenize(ipa_string)
        
        # Count syllables by counting vowels
        syllable_stresses = []
        current_stress = 0
        
        for i, char in enumerate(normalized):
            if char == "ˈ":
                current_stress = 1
            elif char == "ˌ":
                current_stress = 2
        
        # Simple approach: one stress level per vowel nucleus
        for phoneme in phonemes:
            if IPAUtils.is_vowel(phoneme):
                syllable_stresses.append(current_stress)
                current_stress = 0  # Reset after vowel
        
        return "".join(str(s) for s in syllable_stresses)
    
    @staticmethod
    def extract_rhyme_portion(ipa_string: str) -> str:
        """Extract the rhyme portion (from last stressed vowel onward).
        
        Args:
            ipa_string: IPA transcription.
            
        Returns:
            Rhyme portion string.
        """
        normalized = IPAUtils.normalize(ipa_string)
        phonemes = IPAUtils.tokenize(ipa_string)
        
        if not phonemes:
            return ""
        
        # Find the last vowel
        last_vowel_idx = -1
        for i, p in enumerate(phonemes):
            if IPAUtils.is_vowel(p):
                last_vowel_idx = i
        
        if last_vowel_idx == -1:
            return "".join(phonemes)
        
        # Return from last vowel onward
        return "".join(phonemes[last_vowel_idx:])
    
    @staticmethod
    def phonetic_distance(ipa1: str, ipa2: str) -> float:
        """Calculate phonetic distance between two IPA strings.
        
        Uses Levenshtein distance on phoneme sequences normalized
        by the length of the longer sequence.
        
        Args:
            ipa1: First IPA string.
            ipa2: Second IPA string.
            
        Returns:
            Distance between 0.0 (identical) and 1.0 (completely different).
        """
        p1 = IPAUtils.tokenize(ipa1)
        p2 = IPAUtils.tokenize(ipa2)
        
        if not p1 and not p2:
            return 0.0
        if not p1 or not p2:
            return 1.0
        
        # Levenshtein distance
        m, n = len(p1), len(p2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if p1[i-1] == p2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # Deletion
                    dp[i][j-1] + 1,      # Insertion
                    dp[i-1][j-1] + cost, # Substitution
                )
        
        return dp[m][n] / max(m, n)
    
    @staticmethod
    def phonetic_similarity(ipa1: str, ipa2: str) -> float:
        """Calculate phonetic similarity between two IPA strings.
        
        Args:
            ipa1: First IPA string.
            ipa2: Second IPA string.
            
        Returns:
            Similarity between 0.0 (completely different) and 1.0 (identical).
        """
        return 1.0 - IPAUtils.phonetic_distance(ipa1, ipa2)
    
    @staticmethod
    def count_syllables(ipa_string: str) -> int:
        """Count the number of syllables in an IPA string.
        
        Syllables are approximated by counting vowel nuclei.
        
        Args:
            ipa_string: IPA transcription.
            
        Returns:
            Number of syllables.
        """
        phonemes = IPAUtils.tokenize(ipa_string)
        return sum(1 for p in phonemes if IPAUtils.is_vowel(p))
    
    @staticmethod
    def to_ipa_string(phonemes: list[str], add_slashes: bool = True) -> str:
        """Convert phoneme list back to IPA string.
        
        Args:
            phonemes: List of phonemes.
            add_slashes: Whether to add enclosing slashes.
            
        Returns:
            IPA string.
        """
        result = "".join(phonemes)
        if add_slashes:
            return f"/{result}/"
        return result
