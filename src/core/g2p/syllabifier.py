"""
File: src/core/g2p/syllabifier.py
Purpose: Syllable extraction using maximum onset principle per REQ-RHYME-003
"""

from typing import Optional
from src.core.g2p.ipa_utils import IPAUtils, ALL_VOWELS, STRESS_MARKERS


class Syllabifier:
    """Syllabifier using the Maximum Onset Principle.
    
    Splits IPA transcriptions into syllables following the principle
    that consonants are preferably assigned to syllable onsets.
    
    The Maximum Onset Principle states that intervocalic consonants
    should be assigned to the following syllable when forming a
    valid onset cluster.
    """
    
    # Valid English onset clusters
    VALID_ONSETS = {
        # Single consonants
        "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð",
        "s", "z", "ʃ", "ʒ", "h", "m", "n", "l", "r", "w", "j",
        # Two-consonant onsets
        "pl", "pr", "bl", "br", "tr", "dr", "kl", "kr", "gl", "gr",
        "fl", "fr", "θr", "ʃr", "sl", "sm", "sn", "sw", "sp", "st",
        "sk", "sf", "tw", "dw", "kw", "gw", "pj", "bj", "tj", "dj",
        "kj", "gj", "fj", "vj", "θj", "sj", "hj", "mj", "nj", "lj",
        # Three-consonant onsets
        "spl", "spr", "str", "skr", "skw", "skt", "spj", "stj", "skj",
    }
    
    # Sonority hierarchy (higher = more sonorous)
    SONORITY = {
        # Vowels (highest)
        **{v: 10 for v in ALL_VOWELS},
        # Glides
        "j": 8, "w": 8,
        # Liquids
        "l": 7, "r": 7, "ɹ": 7,
        # Nasals
        "m": 6, "n": 6, "ŋ": 6,
        # Fricatives
        "f": 4, "v": 4, "θ": 4, "ð": 4, "s": 4, "z": 4,
        "ʃ": 4, "ʒ": 4, "h": 4,
        # Affricates
        "tʃ": 3, "dʒ": 3,
        # Stops (lowest for consonants)
        "p": 2, "b": 2, "t": 2, "d": 2, "k": 2, "g": 2, "ʔ": 2,
    }
    
    def __init__(self):
        """Initialize syllabifier."""
        pass
    
    def syllabify(self, ipa_string: str) -> list[str]:
        """Split IPA string into syllables.
        
        Args:
            ipa_string: IPA transcription to syllabify.
            
        Returns:
            List of syllable strings.
        """
        phonemes = IPAUtils.tokenize(ipa_string)
        
        if not phonemes:
            return []
        
        # Find vowel positions (syllable nuclei)
        nuclei = []
        for i, p in enumerate(phonemes):
            if IPAUtils.is_vowel(p):
                nuclei.append(i)
        
        if not nuclei:
            # No vowels - entire string is one syllable
            return ["".join(phonemes)]
        
        # Assign consonants to syllables using MOP
        syllables = []
        current_syllable = []
        nucleus_idx = 0
        
        for i, phoneme in enumerate(phonemes):
            current_syllable.append(phoneme)
            
            # Check if we just added a nucleus
            if IPAUtils.is_vowel(phoneme):
                # Look ahead to find onset of next syllable
                if nucleus_idx < len(nuclei) - 1:
                    next_nucleus = nuclei[nucleus_idx + 1]
                    
                    # Get consonants between this nucleus and next
                    coda_onset = phonemes[i+1:next_nucleus]
                    
                    if coda_onset:
                        # Apply Maximum Onset Principle
                        split_point = self._find_onset_split(coda_onset)
                        
                        # Add coda consonants to current syllable
                        for j in range(split_point):
                            current_syllable.append(coda_onset[j])
                        
                        # Finish this syllable
                        syllables.append("".join(current_syllable))
                        current_syllable = []
                        
                        # Start next syllable with onset consonants
                        for j in range(split_point, len(coda_onset)):
                            current_syllable.append(coda_onset[j])
                        
                        # Skip the consonants we already processed
                        # (they'll be added as part of the outer loop iteration)
                        pass
                    else:
                        # No consonants between nuclei
                        syllables.append("".join(current_syllable))
                        current_syllable = []
                
                nucleus_idx += 1
        
        # Handle remaining phonemes
        if current_syllable:
            syllables.append("".join(current_syllable))
        
        return syllables
    
    def _find_onset_split(self, consonants: list[str]) -> int:
        """Find where to split consonant cluster between coda and onset.
        
        Applies Maximum Onset Principle: assigns as many consonants
        as possible to the onset of the following syllable, provided
        they form a valid onset cluster.
        
        Args:
            consonants: List of consonants between two nuclei.
            
        Returns:
            Index where to split (coda ends, onset begins).
        """
        n = len(consonants)
        
        # Try to make onset as long as possible
        for onset_start in range(n):
            potential_onset = "".join(consonants[onset_start:])
            
            # Check if this is a valid onset
            if potential_onset in self.VALID_ONSETS or len(potential_onset) == 0:
                return onset_start
            
            # Check if single consonant at end is valid onset
            if n - onset_start == 1:
                return onset_start
        
        # Default: assign all to coda except last consonant
        return max(0, n - 1)
    
    def get_syllable_structure(self, syllable: str) -> dict:
        """Analyze the structure of a syllable.
        
        Args:
            syllable: Syllable string.
            
        Returns:
            Dict with 'onset', 'nucleus', 'coda' lists.
        """
        phonemes = IPAUtils.tokenize(syllable)
        
        onset = []
        nucleus = []
        coda = []
        
        # Find nucleus (first vowel)
        nucleus_idx = None
        for i, p in enumerate(phonemes):
            if IPAUtils.is_vowel(p):
                nucleus_idx = i
                break
        
        if nucleus_idx is None:
            # No nucleus - unusual but handle it
            return {"onset": phonemes, "nucleus": [], "coda": []}
        
        onset = phonemes[:nucleus_idx]
        
        # Nucleus can include multiple vowels (diphthongs)
        i = nucleus_idx
        while i < len(phonemes) and IPAUtils.is_vowel(phonemes[i]):
            nucleus.append(phonemes[i])
            i += 1
        
        coda = phonemes[i:]
        
        return {"onset": onset, "nucleus": nucleus, "coda": coda}
    
    def get_stress_level(self, syllable: str, ipa_context: str) -> int:
        """Get stress level of a syllable.
        
        Args:
            syllable: Syllable to check.
            ipa_context: Full IPA string for context.
            
        Returns:
            Stress level: 1 (primary), 2 (secondary), 0 (unstressed).
        """
        # Check if syllable is preceded by stress marker in context
        normalized = IPAUtils.normalize(ipa_context)
        syllable_norm = IPAUtils.normalize(syllable)
        
        idx = normalized.find(syllable_norm)
        if idx > 0:
            prev_char = normalized[idx - 1]
            if prev_char == "ˈ":
                return 1
            if prev_char == "ˌ":
                return 2
        
        return 0
    
    def count_syllables(self, ipa_string: str) -> int:
        """Count syllables in an IPA string.
        
        Args:
            ipa_string: IPA transcription.
            
        Returns:
            Number of syllables.
        """
        return len(self.syllabify(ipa_string))
    
    def get_rhyme(self, syllable: str) -> str:
        """Get the rhyme (nucleus + coda) of a syllable.
        
        Args:
            syllable: Syllable string.
            
        Returns:
            Rhyme portion.
        """
        structure = self.get_syllable_structure(syllable)
        return "".join(structure["nucleus"]) + "".join(structure["coda"])
    
    def syllabify_with_stress(self, ipa_string: str) -> list[tuple[str, int]]:
        """Syllabify and include stress information.
        
        Args:
            ipa_string: IPA transcription with stress markers.
            
        Returns:
            List of (syllable, stress_level) tuples.
        """
        syllables = self.syllabify(ipa_string)
        return [
            (s, self.get_stress_level(s, ipa_string))
            for s in syllables
        ]
