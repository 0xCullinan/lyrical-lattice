"""
File: src/detection/phonetic_detectors.py
Purpose: All 13 phonetic-only wordplay device detectors using ARPAbet
"""

import re
import math
from collections import Counter, defaultdict
from typing import Optional

from src.detection.models import (
    HomophoneMatch,
    OronymMatch,
    RhymeMatch,
    AssonanceMatch,
    ConsonanceMatch,
    AlliterationMatch,
    InternalRhymeMatch,
    MultisyllabicRhymeMatch,
    CompoundRhymeMatch,
    OnomatopoeiaMatch,
    SoundTextureMatch,
    StackedRhymeMatch,
)


# =============================================================================
# ARPAbet Phoneme Sets
# =============================================================================

ARPABET_VOWELS = {
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
}

ARPABET_CONSONANTS = {
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH'
}

# Voiced/voiceless pairs for phonetic similarity
VOICED_PAIRS = {
    ('P', 'B'), ('T', 'D'), ('K', 'G'), ('F', 'V'),
    ('TH', 'DH'), ('S', 'Z'), ('SH', 'ZH'), ('CH', 'JH')
}

# Sonority hierarchy for euphony/cacophony (higher = more sonorous)
SONORITY = {
    'AA': 10, 'AE': 10, 'AH': 10, 'AO': 10, 'AW': 10, 'AY': 10,
    'EH': 10, 'ER': 10, 'EY': 10, 'IH': 10, 'IY': 10, 'OW': 10,
    'OY': 10, 'UH': 10, 'UW': 10,  # Vowels
    'W': 8, 'Y': 8,  # Glides
    'L': 7, 'R': 7,  # Liquids
    'M': 6, 'N': 6, 'NG': 6,  # Nasals
    'F': 4, 'V': 4, 'TH': 4, 'DH': 4, 'S': 4, 'Z': 4, 'SH': 4, 'ZH': 4, 'HH': 4,  # Fricatives
    'CH': 3, 'JH': 3,  # Affricates
    'P': 2, 'B': 2, 'T': 2, 'D': 2, 'K': 2, 'G': 2,  # Stops
}

# Harsh consonant clusters for cacophony
HARSH_CLUSTERS = [
    ['K', 'T'], ['K', 'S', 'T'], ['S', 'K'], ['S', 'T', 'R'],
    ['G', 'Z'], ['K', 'S'], ['P', 'T'], ['K', 'T', 'S']
]


def strip_stress(phoneme: str) -> str:
    """Remove stress marker (0, 1, 2) from phoneme"""
    return phoneme.rstrip('012')


def get_stress(phoneme: str) -> int:
    """Get stress level from phoneme (0, 1, or 2)"""
    if phoneme and phoneme[-1].isdigit():
        return int(phoneme[-1])
    return 0


def is_vowel(phoneme: str) -> bool:
    """Check if phoneme is a vowel"""
    return strip_stress(phoneme) in ARPABET_VOWELS


def is_consonant(phoneme: str) -> bool:
    """Check if phoneme is a consonant"""
    return strip_stress(phoneme) in ARPABET_CONSONANTS


# =============================================================================
# 1. HOMOPHONE DETECTOR
# =============================================================================

class HomophoneDetector:
    """Detect homophones - words with identical phonemes, different spellings.

    O(1) lookup using reverse index.
    """

    def __init__(self, reverse_index: dict[str, list[str]]):
        """
        Args:
            reverse_index: Mapping of phoneme string → list of words
                           e.g., "P EY1 R" → ["pair", "pear", "pare"]
        """
        self.reverse_index = reverse_index

    def detect(self, phoneme_sequence: list[str]) -> list[HomophoneMatch]:
        """Detect homophones for a phoneme sequence.

        Args:
            phoneme_sequence: List of ARPAbet phonemes

        Returns:
            List of homophone matches (empty if no homophones)
        """
        key = " ".join(phoneme_sequence)

        if key not in self.reverse_index:
            return []

        words = self.reverse_index[key]

        if len(words) < 2:
            return []  # Need at least 2 words for homophone

        return [HomophoneMatch(
            phonemes=key,
            words=words,
            confidence=1.0,
            type="exact_homophone"
        )]

    def detect_in_bar(self, bar_phonemes: list[list[str]]) -> list[HomophoneMatch]:
        """Detect homophones for each word in a bar.

        Args:
            bar_phonemes: List of phoneme sequences, one per word

        Returns:
            All homophone matches found
        """
        results = []
        for word_phonemes in bar_phonemes:
            matches = self.detect(word_phonemes)
            results.extend(matches)
        return results


# =============================================================================
# 2. ORONYM DETECTOR
# =============================================================================

class OronymDetector:
    """Detect oronyms - phrases that sound identical with different word boundaries.

    Uses beam search for efficient segmentation.
    """

    def __init__(
        self,
        reverse_index: dict[str, list[str]],
        word_frequencies: Optional[dict[str, int]] = None,
        beam_width: int = 10,
        max_words: int = 5
    ):
        """
        Args:
            reverse_index: Mapping of phoneme string → list of words
            word_frequencies: Optional word frequency dict for scoring
            beam_width: Beam search width
            max_words: Maximum words in a segmentation
        """
        self.reverse_index = reverse_index
        self.word_frequencies = word_frequencies or {}
        self.beam_width = beam_width
        self.max_words = max_words

    def detect(self, phoneme_sequence: list[str]) -> list[OronymMatch]:
        """Find all valid word segmentations of phoneme sequence.

        Uses dynamic programming with beam search pruning.

        Args:
            phoneme_sequence: List of ARPAbet phonemes

        Returns:
            List of oronym matches ranked by score
        """
        n = len(phoneme_sequence)
        if n == 0:
            return []

        # dp[i] = list of (words, score) ending at position i
        dp: list[list[tuple[list[str], float]]] = [[] for _ in range(n + 1)]
        dp[0] = [([], 1.0)]

        for i in range(n):
            if not dp[i]:
                continue

            # Try all window sizes from position i
            for j in range(i + 1, min(i + 15, n + 1)):  # Max 15 phonemes per word
                window = " ".join(phoneme_sequence[i:j])

                if window in self.reverse_index:
                    words_for_window = self.reverse_index[window]

                    for prev_words, prev_score in dp[i]:
                        if len(prev_words) < self.max_words:
                            # Take top 3 most common words for this phoneme sequence
                            for word in words_for_window[:3]:
                                new_path = prev_words + [word]
                                score = prev_score * self._word_score(word)
                                dp[j].append((new_path, score))

            # Prune beam at each position
            dp[i + 1] = sorted(dp[i + 1], key=lambda x: -x[1])[:self.beam_width]

        # Collect complete segmentations
        results = []
        for words, score in dp[n]:
            if len(words) >= 1:
                results.append(OronymMatch(
                    original_phonemes=phoneme_sequence,
                    segmentation=words,
                    score=score,
                    confidence=min(score, 1.0),
                    type="oronym"
                ))

        return sorted(results, key=lambda x: -x.score)[:10]

    def _word_score(self, word: str) -> float:
        """Score based on word frequency (common words score higher)."""
        freq = self.word_frequencies.get(word.lower(), 1)
        return math.log(freq + 1) / math.log(1000000)


# =============================================================================
# 3. PERFECT RHYME DETECTOR
# =============================================================================

class PerfectRhymeDetector:
    """Detect perfect rhymes - words matching from last stressed vowel to end."""

    def __init__(self, reverse_index: dict[str, list[str]]):
        """
        Args:
            reverse_index: Mapping of phoneme string → list of words
        """
        self.reverse_index = reverse_index
        self.rhyme_index = self._build_rhyme_index()

    def _build_rhyme_index(self) -> dict[str, list[str]]:
        """Pre-compute rhyme portions for O(1) lookup."""
        rhyme_idx: dict[str, list[str]] = defaultdict(list)

        for phonemes_str, words in self.reverse_index.items():
            phonemes = phonemes_str.split()
            rhyme = self.extract_rhyme_portion(phonemes)
            if rhyme:
                rhyme_key = " ".join(rhyme)
                rhyme_idx[rhyme_key].extend(words)

        return dict(rhyme_idx)

    def extract_rhyme_portion(self, phonemes: list[str]) -> list[str]:
        """Extract from last stressed vowel onward.

        Args:
            phonemes: List of ARPAbet phonemes

        Returns:
            Rhyme portion (phonemes from stressed vowel to end)
        """
        last_stressed_idx = -1

        for i, phone in enumerate(phonemes):
            base = strip_stress(phone)
            stress = get_stress(phone)

            if base in ARPABET_VOWELS and stress in (1, 2):
                last_stressed_idx = i

        if last_stressed_idx == -1:
            # No stressed vowel, use last vowel
            for i in range(len(phonemes) - 1, -1, -1):
                if strip_stress(phonemes[i]) in ARPABET_VOWELS:
                    last_stressed_idx = i
                    break

        if last_stressed_idx == -1:
            return []

        return phonemes[last_stressed_idx:]

    def detect(self, phonemes: list[str]) -> list[RhymeMatch]:
        """Find all perfect rhymes for given phonemes.

        Args:
            phonemes: List of ARPAbet phonemes

        Returns:
            List of rhyme matches
        """
        rhyme = self.extract_rhyme_portion(phonemes)
        if not rhyme:
            return []

        rhyme_key = " ".join(rhyme)
        matches = self.rhyme_index.get(rhyme_key, [])

        if not matches:
            return []

        return [RhymeMatch(
            query_phonemes=phonemes,
            rhyme_portion=rhyme,
            matches=matches,
            rhyme_type="perfect",
            confidence=1.0
        )]


# =============================================================================
# 4. SLANT/NEAR RHYME DETECTOR
# =============================================================================

class SlantRhymeDetector:
    """Detect slant/near rhymes - phonetically similar but not identical."""

    def __init__(
        self,
        rhyme_index: dict[str, list[str]],
        similarity_threshold: float = 0.7
    ):
        """
        Args:
            rhyme_index: Pre-built rhyme portion → words index
            similarity_threshold: Minimum similarity for slant rhyme (0-1)
        """
        self.rhyme_index = rhyme_index
        self.similarity_threshold = similarity_threshold
        self._rhyme_detector = PerfectRhymeDetector.__new__(PerfectRhymeDetector)
        self._rhyme_detector.rhyme_index = rhyme_index

    def phoneme_similarity(self, p1: list[str], p2: list[str]) -> float:
        """Calculate normalized Levenshtein similarity between phoneme sequences.

        Args:
            p1, p2: Lists of ARPAbet phonemes

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if not p1 or not p2:
            return 0.0

        m, n = len(p1), len(p2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                phone1 = strip_stress(p1[i - 1])
                phone2 = strip_stress(p2[j - 1])

                if phone1 == phone2:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    sub_cost = self._substitution_cost(phone1, phone2)
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # deletion
                        dp[i][j - 1] + 1,  # insertion
                        dp[i - 1][j - 1] + sub_cost  # substitution
                    )

        distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len)

    def _substitution_cost(self, p1: str, p2: str) -> float:
        """Calculate substitution cost based on phonetic similarity.

        Similar sounds cost less to substitute.
        """
        # Voiced/voiceless pairs: 0.3
        if (p1, p2) in VOICED_PAIRS or (p2, p1) in VOICED_PAIRS:
            return 0.3

        # Same phoneme class: 0.5
        if self._same_class(p1, p2):
            return 0.5

        # Different class: 1.0
        return 1.0

    def _same_class(self, p1: str, p2: str) -> bool:
        """Check if two phonemes are in the same class."""
        v1 = p1 in ARPABET_VOWELS
        v2 = p2 in ARPABET_VOWELS
        if v1 and v2:
            return True
        if not v1 and not v2:
            s1 = SONORITY.get(p1, 5)
            s2 = SONORITY.get(p2, 5)
            return abs(s1 - s2) <= 2
        return False

    def extract_rhyme_portion(self, phonemes: list[str]) -> list[str]:
        """Extract rhyme portion from phonemes."""
        last_stressed_idx = -1

        for i, phone in enumerate(phonemes):
            base = strip_stress(phone)
            stress = get_stress(phone)

            if base in ARPABET_VOWELS and stress in (1, 2):
                last_stressed_idx = i

        if last_stressed_idx == -1:
            for i in range(len(phonemes) - 1, -1, -1):
                if strip_stress(phonemes[i]) in ARPABET_VOWELS:
                    last_stressed_idx = i
                    break

        if last_stressed_idx == -1:
            return []

        return phonemes[last_stressed_idx:]

    def detect(self, phonemes: list[str], top_k: int = 20) -> list[RhymeMatch]:
        """Find slant rhymes by comparing to all rhyme portions.

        Args:
            phonemes: List of ARPAbet phonemes
            top_k: Maximum number of results

        Returns:
            List of slant rhyme matches ranked by similarity
        """
        query_rhyme = self.extract_rhyme_portion(phonemes)
        if not query_rhyme:
            return []

        results = []

        for rhyme_key, words in self.rhyme_index.items():
            candidate_rhyme = rhyme_key.split()

            sim = self.phoneme_similarity(query_rhyme, candidate_rhyme)

            # Include if similar but not perfect match
            if self.similarity_threshold <= sim < 1.0:
                results.append(RhymeMatch(
                    query_phonemes=phonemes,
                    rhyme_portion=query_rhyme,
                    matches=words,
                    rhyme_type="slant",
                    confidence=sim
                ))

        return sorted(results, key=lambda x: -x.confidence)[:top_k]


# =============================================================================
# 5. ASSONANCE DETECTOR
# =============================================================================

class AssonanceDetector:
    """Detect assonance - repetition of vowel sounds."""

    def extract_vowels(self, phonemes: list[str]) -> list[str]:
        """Extract vowel sequence from phonemes."""
        return [strip_stress(p) for p in phonemes if is_vowel(p)]

    def detect(
        self,
        bar_phonemes: list[list[str]],
        min_count: int = 3
    ) -> list[AssonanceMatch]:
        """Detect repeated vowel patterns in a bar.

        Args:
            bar_phonemes: List of phoneme sequences, one per word
            min_count: Minimum repetitions to count as assonance

        Returns:
            List of assonance matches
        """
        all_vowels = []
        word_vowel_map = []  # Track which word each vowel came from

        for word_idx, word_phones in enumerate(bar_phonemes):
            word_vowels = self.extract_vowels(word_phones)
            for v in word_vowels:
                all_vowels.append(v)
                word_vowel_map.append(word_idx)

        vowel_counts = Counter(all_vowels)

        results = []
        for vowel, count in vowel_counts.items():
            if count >= min_count:
                words_with_vowel = set()
                for i, v in enumerate(all_vowels):
                    if v == vowel:
                        words_with_vowel.add(word_vowel_map[i])

                results.append(AssonanceMatch(
                    vowel=vowel,
                    count=count,
                    word_indices=list(words_with_vowel),
                    confidence=min(count / 5.0, 1.0),
                    type="assonance"
                ))

        return sorted(results, key=lambda x: -x.count)


# =============================================================================
# 6. CONSONANCE DETECTOR
# =============================================================================

class ConsonanceDetector:
    """Detect consonance - repetition of consonant sounds."""

    def extract_consonants(self, phonemes: list[str]) -> list[str]:
        """Extract consonant sequence from phonemes."""
        return [strip_stress(p) for p in phonemes if is_consonant(p)]

    def detect(
        self,
        bar_phonemes: list[list[str]],
        min_count: int = 3
    ) -> list[ConsonanceMatch]:
        """Detect repeated consonant patterns.

        Args:
            bar_phonemes: List of phoneme sequences, one per word
            min_count: Minimum repetitions to count as consonance

        Returns:
            List of consonance matches
        """
        all_consonants = []
        word_map = []

        for word_idx, word_phones in enumerate(bar_phonemes):
            word_cons = self.extract_consonants(word_phones)
            for c in word_cons:
                all_consonants.append(c)
                word_map.append(word_idx)

        cons_counts = Counter(all_consonants)

        results = []
        for consonant, count in cons_counts.items():
            if count >= min_count:
                words_with_cons = set(
                    word_map[i] for i, c in enumerate(all_consonants) if c == consonant
                )

                results.append(ConsonanceMatch(
                    consonant=consonant,
                    count=count,
                    word_indices=list(words_with_cons),
                    confidence=min(count / 5.0, 1.0),
                    type="consonance"
                ))

        return sorted(results, key=lambda x: -x.count)


# =============================================================================
# 7. ALLITERATION DETECTOR
# =============================================================================

class AlliterationDetector:
    """Detect alliteration - repetition of initial consonant sounds."""

    def get_initial_consonant(self, phonemes: list[str]) -> Optional[str]:
        """Get first consonant of word."""
        if not phonemes:
            return None

        first = strip_stress(phonemes[0])

        if first in ARPABET_CONSONANTS:
            return first

        return None  # Word starts with vowel

    def detect(
        self,
        bar_phonemes: list[list[str]],
        min_run: int = 2
    ) -> list[AlliterationMatch]:
        """Detect consecutive words with same initial consonant.

        Args:
            bar_phonemes: List of phoneme sequences, one per word
            min_run: Minimum consecutive words to count as alliteration

        Returns:
            List of alliteration matches
        """
        initials = [self.get_initial_consonant(wp) for wp in bar_phonemes]

        results = []
        i = 0

        while i < len(initials):
            if initials[i] is None:
                i += 1
                continue

            run_start = i
            run_consonant = initials[i]

            while i < len(initials) and initials[i] == run_consonant:
                i += 1

            run_length = i - run_start

            if run_length >= min_run:
                results.append(AlliterationMatch(
                    consonant=run_consonant,
                    word_indices=list(range(run_start, i)),
                    run_length=run_length,
                    confidence=min(run_length / 4.0, 1.0),
                    type="alliteration"
                ))

        return results


# =============================================================================
# 8. INTERNAL RHYME DETECTOR
# =============================================================================

class InternalRhymeDetector:
    """Detect internal rhyme - rhyme within a line (not at end)."""

    def __init__(self, rhyme_detector: PerfectRhymeDetector):
        self.rhyme_detector = rhyme_detector

    def detect(self, line_phonemes: list[list[str]]) -> list[InternalRhymeMatch]:
        """Find rhyming pairs within a line (excluding final word).

        Args:
            line_phonemes: List of phoneme sequences for each word in line

        Returns:
            List of internal rhyme matches
        """
        n = len(line_phonemes)
        if n < 3:
            return []

        results = []

        # Compare all pairs except the last word
        for i in range(n - 1):
            rhyme_i = self.rhyme_detector.extract_rhyme_portion(line_phonemes[i])
            if not rhyme_i:
                continue

            for j in range(i + 1, n - 1):
                rhyme_j = self.rhyme_detector.extract_rhyme_portion(line_phonemes[j])
                if not rhyme_j:
                    continue

                if rhyme_i == rhyme_j:
                    results.append(InternalRhymeMatch(
                        word_indices=(i, j),
                        rhyme_portion=rhyme_i,
                        type="internal_rhyme",
                        confidence=1.0
                    ))

        return results


# =============================================================================
# 9. MULTISYLLABIC RHYME DETECTOR
# =============================================================================

class MultisyllabicRhymeDetector:
    """Detect multisyllabic rhymes - 2+ syllables matching."""

    def __init__(self, reverse_index: dict[str, list[str]]):
        """
        Args:
            reverse_index: Mapping of phoneme string → list of words
        """
        self.reverse_index = reverse_index
        self.multisyl_index = self._build_multisyl_index()

    def _syllabify_arpabet(self, phonemes: list[str]) -> list[str]:
        """Simple ARPAbet syllabification based on vowels.

        Each vowel starts a new syllable.
        """
        syllables = []
        current = []

        for phone in phonemes:
            current.append(phone)
            if is_vowel(phone):
                syllables.append(" ".join(current))
                current = []

        # Handle trailing consonants
        if current and syllables:
            syllables[-1] += " " + " ".join(current)
        elif current:
            syllables.append(" ".join(current))

        return syllables

    def _build_multisyl_index(self) -> dict[int, dict[str, list[str]]]:
        """Index words by their last 2, 3, 4 syllables."""
        index: dict[int, dict[str, list[str]]] = {
            2: defaultdict(list),
            3: defaultdict(list),
            4: defaultdict(list)
        }

        for phonemes_str, words in self.reverse_index.items():
            phonemes = phonemes_str.split()
            syllables = self._syllabify_arpabet(phonemes)

            for n in [2, 3, 4]:
                if len(syllables) >= n:
                    key = " | ".join(syllables[-n:])
                    index[n][key].extend(words)

        return {k: dict(v) for k, v in index.items()}

    def detect(
        self,
        phonemes: list[str],
        min_syllables: int = 2
    ) -> list[MultisyllabicRhymeMatch]:
        """Find words that rhyme on 2+ syllables.

        Args:
            phonemes: List of ARPAbet phonemes
            min_syllables: Minimum syllables to match

        Returns:
            List of multisyllabic rhyme matches
        """
        syllables = self._syllabify_arpabet(phonemes)

        results = []

        for n in range(min(len(syllables), 4), min_syllables - 1, -1):
            key = " | ".join(syllables[-n:])

            if key in self.multisyl_index.get(n, {}):
                matches = self.multisyl_index[n][key]

                results.append(MultisyllabicRhymeMatch(
                    syllables_matched=n,
                    matching_syllables=syllables[-n:],
                    matches=matches,
                    confidence=min(n / 3.0, 1.0),
                    type=f"{n}_syllable_rhyme"
                ))

        return results


# =============================================================================
# 10. COMPOUND/MOSAIC RHYME DETECTOR
# =============================================================================

class CompoundRhymeDetector:
    """Detect compound rhymes - multi-word phrases rhyming with single words."""

    def __init__(self, rhyme_detector: PerfectRhymeDetector):
        self.rhyme_detector = rhyme_detector

    def detect(
        self,
        multi_word_phonemes: list[list[str]]
    ) -> list[CompoundRhymeMatch]:
        """Find single words that rhyme with multi-word phrases.

        Args:
            multi_word_phonemes: List of phoneme sequences for each word

        Returns:
            List of compound rhyme matches
        """
        # Concatenate phonemes from multiple words
        concat = []
        for word_phones in multi_word_phonemes:
            concat.extend(word_phones)

        # Extract rhyme portion from concatenated phrase
        rhyme = self.rhyme_detector.extract_rhyme_portion(concat)
        if not rhyme:
            return []

        rhyme_key = " ".join(rhyme)

        # Find single words with matching rhyme
        matches = self.rhyme_detector.rhyme_index.get(rhyme_key, [])

        if matches:
            return [CompoundRhymeMatch(
                multi_word_phrase=multi_word_phonemes,
                phrase_rhyme=rhyme,
                single_word_matches=matches,
                type="compound_rhyme",
                confidence=1.0
            )]

        return []


# =============================================================================
# 11. ONOMATOPOEIA DETECTOR
# =============================================================================

class OnomatopoeiaDetector:
    """Detect onomatopoeia - words that phonetically imitate sounds."""

    ONOMATOPOEIA_LEXICON = {
        # Animal sounds
        "buzz", "hiss", "meow", "woof", "chirp", "roar", "growl",
        "moo", "oink", "quack", "bark", "purr", "howl", "squeak",
        # Impact sounds
        "bang", "crash", "thud", "smash", "crack", "pop", "snap",
        "boom", "thump", "clang", "clatter", "crunch", "slam",
        # Water sounds
        "splash", "drip", "gurgle", "slosh", "whoosh", "swoosh",
        "splatter", "gush", "trickle", "bubble",
        # Voice sounds
        "whisper", "murmur", "giggle", "groan", "sigh", "gasp",
        "shriek", "scream", "mumble", "grunt", "snore", "cough",
        # Mechanical
        "beep", "click", "buzz", "whir", "zoom", "vroom", "honk",
        "screech", "squeal", "rattle", "clunk",
        # Misc
        "pow", "zap", "zing", "fizz", "sizzle", "crackle", "rustle",
        "swish", "whack", "thwack", "splat", "plop", "clap",
    }

    # Phonetic patterns associated with onomatopoeia
    SOUND_PATTERNS = {
        "sibilant_end": re.compile(r".*\b(S|Z)\b$"),  # hiss, buzz, fizz
        "plosive_start": re.compile(r"^\b(P|T|B|D|K|G)\b.*"),  # bang, pop, crash
        "nasal_vibrant": re.compile(r".*\b(M|N)\b.*\b(M|N)\b.*"),  # murmur, hum
    }

    def detect(self, word: str, phonemes: list[str]) -> Optional[OnomatopoeiaMatch]:
        """Check if word is onomatopoeia.

        Args:
            word: The word text
            phonemes: ARPAbet phoneme sequence

        Returns:
            OnomatopoeiaMatch if detected, None otherwise
        """
        word_lower = word.lower()

        # Direct lexicon lookup
        if word_lower in self.ONOMATOPOEIA_LEXICON:
            return OnomatopoeiaMatch(
                word=word,
                phonemes=phonemes,
                type="onomatopoeia",
                confidence=1.0,
                category=self._categorize(word_lower)
            )

        # Pattern-based detection
        phoneme_str = " ".join(strip_stress(p) for p in phonemes)
        for pattern_name, pattern in self.SOUND_PATTERNS.items():
            if pattern.match(phoneme_str):
                return OnomatopoeiaMatch(
                    word=word,
                    phonemes=phonemes,
                    type="potential_onomatopoeia",
                    confidence=0.5,
                    pattern=pattern_name
                )

        return None

    def _categorize(self, word: str) -> str:
        """Categorize onomatopoeia by type."""
        animals = {"buzz", "hiss", "meow", "woof", "chirp", "roar", "growl",
                   "moo", "oink", "quack", "bark", "purr", "howl", "squeak"}
        impacts = {"bang", "crash", "thud", "smash", "crack", "pop", "snap",
                   "boom", "thump", "clang", "clatter", "crunch", "slam"}
        water = {"splash", "drip", "gurgle", "slosh", "whoosh", "swoosh",
                 "splatter", "gush", "trickle", "bubble"}
        voice = {"whisper", "murmur", "giggle", "groan", "sigh", "gasp",
                 "shriek", "scream", "mumble", "grunt", "snore", "cough"}
        mechanical = {"beep", "click", "whir", "zoom", "vroom", "honk",
                      "screech", "squeal", "rattle", "clunk"}

        if word in animals:
            return "animal"
        if word in impacts:
            return "impact"
        if word in water:
            return "water"
        if word in voice:
            return "voice"
        if word in mechanical:
            return "mechanical"
        return "misc"


# =============================================================================
# 12. EUPHONY/CACOPHONY DETECTOR
# =============================================================================

class EuphonyCacophonyDetector:
    """Detect euphony (pleasant) or cacophony (harsh) sound combinations."""

    def calculate_euphony(self, phonemes: list[str]) -> float:
        """Calculate euphony score (0-1, higher = more pleasant).

        Args:
            phonemes: List of ARPAbet phonemes

        Returns:
            Euphony score
        """
        if len(phonemes) < 2:
            return 0.5

        total_score = 0

        for i in range(len(phonemes) - 1):
            p1 = strip_stress(phonemes[i])
            p2 = strip_stress(phonemes[i + 1])

            s1 = SONORITY.get(p1, 5)
            s2 = SONORITY.get(p2, 5)

            # Smooth transitions score higher
            transition_smoothness = 1.0 - abs(s1 - s2) / 10.0
            total_score += (s1 + s2) / 20.0 * transition_smoothness

        return total_score / (len(phonemes) - 1)

    def calculate_cacophony(self, phonemes: list[str]) -> float:
        """Calculate cacophony score (0-1, higher = harsher).

        Args:
            phonemes: List of ARPAbet phonemes

        Returns:
            Cacophony score
        """
        harsh_count = 0

        phoneme_bases = [strip_stress(p) for p in phonemes]

        # Check for harsh clusters
        for cluster in HARSH_CLUSTERS:
            cluster_str = ''.join(cluster)
            phoneme_str = ''.join(phoneme_bases)
            if cluster_str in phoneme_str:
                harsh_count += 1

        # Count consecutive stops/fricatives
        for i in range(len(phoneme_bases) - 1):
            s1 = SONORITY.get(phoneme_bases[i], 5)
            s2 = SONORITY.get(phoneme_bases[i + 1], 5)
            if s1 <= 4 and s2 <= 4:
                harsh_count += 0.5

        return min(harsh_count / 3.0, 1.0)

    def detect(self, phonemes: list[str]) -> SoundTextureMatch:
        """Classify phrase as euphonic or cacophonic.

        Args:
            phonemes: List of ARPAbet phonemes

        Returns:
            SoundTextureMatch with classification
        """
        euphony = self.calculate_euphony(phonemes)
        cacophony = self.calculate_cacophony(phonemes)

        if euphony > 0.7 and cacophony < 0.3:
            return SoundTextureMatch(
                type="euphony",
                euphony_score=euphony,
                cacophony_score=cacophony,
                confidence=euphony
            )
        elif cacophony > 0.5:
            return SoundTextureMatch(
                type="cacophony",
                euphony_score=euphony,
                cacophony_score=cacophony,
                confidence=cacophony
            )
        else:
            return SoundTextureMatch(
                type="neutral",
                euphony_score=euphony,
                cacophony_score=cacophony,
                confidence=0.5
            )


# =============================================================================
# 13. STACKED RHYME DETECTOR
# =============================================================================

class StackedRhymeDetector:
    """Detect stacked rhymes - high density of rhymes within a bar."""

    def __init__(self, rhyme_detector: PerfectRhymeDetector):
        self.rhyme_detector = rhyme_detector

    def detect(
        self,
        bar_phonemes: list[list[str]],
        threshold: int = 3
    ) -> Optional[StackedRhymeMatch]:
        """Detect high rhyme density in a bar.

        Args:
            bar_phonemes: List of phoneme sequences, one per word
            threshold: Minimum total rhymes to qualify as stacked

        Returns:
            StackedRhymeMatch if threshold met, None otherwise
        """
        # Extract rhyme portions for all words
        rhymes = []
        for i, word_phones in enumerate(bar_phonemes):
            rhyme = self.rhyme_detector.extract_rhyme_portion(word_phones)
            if rhyme:
                rhymes.append((i, " ".join(rhyme)))

        # Group by rhyme
        rhyme_groups: dict[str, list[int]] = defaultdict(list)
        for idx, rhyme_key in rhymes:
            rhyme_groups[rhyme_key].append(idx)

        # Find largest rhyme group
        largest_group = max(rhyme_groups.values(), key=len, default=[])

        # Count total rhyming words (groups of 2+)
        total_rhymes = sum(
            len(group) for group in rhyme_groups.values() if len(group) >= 2
        )

        if total_rhymes >= threshold:
            return StackedRhymeMatch(
                rhyme_groups=dict(rhyme_groups),
                total_rhymes=total_rhymes,
                largest_group_size=len(largest_group),
                density=total_rhymes / len(bar_phonemes) if bar_phonemes else 0,
                confidence=min(total_rhymes / 5.0, 1.0),
                type="stacked_rhyme"
            )

        return None
