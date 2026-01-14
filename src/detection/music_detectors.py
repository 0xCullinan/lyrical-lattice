"""
File: src/detection/music_detectors.py
Purpose: 4 music-specific wordplay device detectors
"""

from collections import defaultdict
from typing import Optional

import numpy as np

from src.detection.models import (
    PolyrhythmicMatch,
    BreathRhymeMatch,
    MelismaMatch,
    SampleFlipMatch,
)
from src.detection.phonetic_detectors import (
    PerfectRhymeDetector,
    strip_stress,
)


# =============================================================================
# 18. POLYRHYTHMIC RHYME DETECTOR
# =============================================================================

class PolyrhythmicRhymeDetector:
    """Detect polyrhythmic rhymes - rhymes on varying beat positions.

    Polyrhythmic rhyme occurs when rhyming words are placed at
    different metric positions across multiple bars, creating
    rhythmic interest beyond simple end-rhymes.
    """

    def __init__(
        self,
        rhyme_detector: PerfectRhymeDetector,
        bpm: int = 90
    ):
        """
        Args:
            rhyme_detector: For extracting rhyme portions
            bpm: Beats per minute of the track
        """
        self.rhyme_detector = rhyme_detector
        self.bpm = bpm
        self.beats_per_bar = 4

    def detect(
        self,
        bars: list[list[tuple[list[str], float]]]
    ) -> Optional[PolyrhythmicMatch]:
        """Detect rhymes on varying beat positions across bars.

        Args:
            bars: List of bars, each containing list of (phonemes, timestamp) tuples

        Returns:
            PolyrhythmicMatch if detected, None otherwise
        """
        # Extract rhyme positions in beat space
        rhyme_positions: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for bar_idx, bar in enumerate(bars):
            if not bar:
                continue

            bar_start_time = bar[0][1]

            for phonemes, timestamp in bar:
                rhyme = self.rhyme_detector.extract_rhyme_portion(phonemes)
                if not rhyme:
                    continue

                # Convert timestamp to beat position (0-3.99)
                relative_time = timestamp - bar_start_time
                beat_duration = 60.0 / self.bpm
                beat_position = (relative_time / beat_duration) % self.beats_per_bar

                rhyme_key = " ".join(rhyme)
                rhyme_positions[rhyme_key].append((bar_idx, beat_position))

        # Find rhyme sets with varying positions
        for rhyme_key, positions in rhyme_positions.items():
            if len(positions) < 3:
                continue

            beat_positions = [p[1] for p in positions]
            position_variance = float(np.var(beat_positions))

            # High variance = polyrhythmic
            if position_variance > 0.5:
                return PolyrhythmicMatch(
                    rhyme=rhyme_key,
                    positions=positions,
                    variance=position_variance,
                    pattern=self._classify_pattern(beat_positions),
                    confidence=min(position_variance, 1.0),
                    type="polyrhythmic_rhyme"
                )

        return None

    def _classify_pattern(self, positions: list[float]) -> str:
        """Classify the rhythmic pattern.

        Args:
            positions: List of beat positions

        Returns:
            Pattern classification string
        """
        # Quantize to 8th notes
        rounded = [round(p * 2) / 2 for p in positions]
        unique = len(set(rounded))

        if unique == len(rounded):
            return "fully_varied"
        elif unique >= len(rounded) * 0.7:
            return "mostly_varied"
        else:
            return "partially_varied"

    def analyze_rhythmic_density(
        self,
        bars: list[list[tuple[list[str], float]]]
    ) -> dict:
        """Analyze rhythmic density of rhymes across bars.

        Args:
            bars: List of bars with phonemes and timestamps

        Returns:
            Analysis dictionary with density metrics
        """
        total_rhymes = 0
        beat_distribution = defaultdict(int)

        for bar in bars:
            if not bar:
                continue

            bar_start = bar[0][1]

            for phonemes, timestamp in bar:
                rhyme = self.rhyme_detector.extract_rhyme_portion(phonemes)
                if rhyme:
                    total_rhymes += 1
                    beat_duration = 60.0 / self.bpm
                    beat_pos = int((timestamp - bar_start) / beat_duration) % 4
                    beat_distribution[beat_pos] += 1

        return {
            "total_rhymes": total_rhymes,
            "beat_distribution": dict(beat_distribution),
            "avg_per_bar": total_rhymes / len(bars) if bars else 0,
        }


# =============================================================================
# 19. BREATH RHYME DETECTOR
# =============================================================================

class BreathRhymeDetector:
    """Detect breath rhymes - rhymes at natural pause points.

    Breath rhymes occur at natural breathing/pause points in the
    flow, typically at mid-bar and end-bar positions.
    """

    # Typical breath points as beat fractions
    BREATH_POINTS = [2.0, 4.0]  # Mid-bar (beat 2) and end-bar (beat 4)
    TOLERANCE = 0.3  # ± beats tolerance

    def __init__(self, rhyme_detector: PerfectRhymeDetector):
        """
        Args:
            rhyme_detector: For extracting rhyme portions
        """
        self.rhyme_detector = rhyme_detector

    def detect(
        self,
        bar_phonemes_with_timing: list[tuple[list[str], float]],
        bpm: int = 90
    ) -> list[BreathRhymeMatch]:
        """Detect rhymes at breath points.

        Args:
            bar_phonemes_with_timing: List of (phonemes, timestamp) tuples
            bpm: Beats per minute

        Returns:
            List of breath rhyme matches
        """
        beat_duration = 60.0 / bpm

        results = []

        for phonemes, timestamp in bar_phonemes_with_timing:
            # Get position in beat units
            beat_pos = (timestamp / beat_duration) % 4

            # Check if near a breath point
            for bp in self.BREATH_POINTS:
                if abs(beat_pos - bp) < self.TOLERANCE:
                    rhyme = self.rhyme_detector.extract_rhyme_portion(phonemes)
                    if rhyme:
                        results.append(BreathRhymeMatch(
                            phonemes=phonemes,
                            rhyme_portion=rhyme,
                            beat_position=beat_pos,
                            breath_point=bp,
                            type="breath_rhyme"
                        ))
                    break

        return results

    def find_breath_rhyme_pairs(
        self,
        bars: list[list[tuple[list[str], float]]],
        bpm: int = 90
    ) -> list[tuple[BreathRhymeMatch, BreathRhymeMatch]]:
        """Find pairs of breath rhymes that rhyme with each other.

        Args:
            bars: List of bars with phonemes and timestamps
            bpm: Beats per minute

        Returns:
            List of rhyming breath point pairs
        """
        all_breath_rhymes = []

        for bar in bars:
            matches = self.detect(bar, bpm)
            all_breath_rhymes.extend(matches)

        # Group by rhyme portion
        rhyme_groups: dict[str, list[BreathRhymeMatch]] = defaultdict(list)
        for match in all_breath_rhymes:
            key = " ".join(match.rhyme_portion)
            rhyme_groups[key].append(match)

        # Find pairs
        pairs = []
        for rhyme_key, matches in rhyme_groups.items():
            if len(matches) >= 2:
                for i in range(len(matches)):
                    for j in range(i + 1, len(matches)):
                        pairs.append((matches[i], matches[j]))

        return pairs


# =============================================================================
# 20. MELISMA WORDPLAY DETECTOR
# =============================================================================

class MelismaDetector:
    """Detect melisma wordplay - stretched syllables revealing hidden words.

    Melisma is when a singer stretches a syllable over multiple notes.
    Sometimes this stretching can reveal or suggest hidden words.
    """

    def __init__(self, reverse_index: dict[str, list[str]]):
        """
        Args:
            reverse_index: Phoneme → words mapping
        """
        self.reverse_index = reverse_index

    def detect(
        self,
        word_phonemes: list[str],
        stretched_phonemes: list[str]
    ) -> Optional[MelismaMatch]:
        """Detect if stretched singing reveals hidden word.

        Args:
            word_phonemes: Original word phonemes
            stretched_phonemes: Phonemes as sung (may have repetitions)

        Returns:
            MelismaMatch if hidden word found
        """
        # Collapse repeated phonemes
        collapsed = []
        for p in stretched_phonemes:
            if not collapsed or collapsed[-1] != p:
                collapsed.append(p)

        # Check if stretched version matches a different word
        stretched_key = " ".join(stretched_phonemes)

        if stretched_key in self.reverse_index:
            hidden_words = self.reverse_index[stretched_key]

            # Filter out the original word
            original_key = " ".join(word_phonemes)
            original_words = self.reverse_index.get(original_key, [])

            new_meanings = [w for w in hidden_words if w not in original_words]

            if new_meanings:
                return MelismaMatch(
                    original_phonemes=word_phonemes,
                    stretched_phonemes=stretched_phonemes,
                    revealed_words=new_meanings,
                    type="melisma_wordplay",
                    confidence=0.7
                )

        return None

    def detect_repeated_vowels(
        self,
        word: str,
        word_phonemes: list[str],
        repetition_count: int = 2
    ) -> Optional[MelismaMatch]:
        """Detect potential melisma by simulating vowel repetition.

        Args:
            word: Original word text
            word_phonemes: Original phonemes
            repetition_count: How many times to repeat each vowel

        Returns:
            MelismaMatch if hidden word found
        """
        from src.detection.phonetic_detectors import is_vowel

        # Create stretched version by repeating vowels
        stretched = []
        for p in word_phonemes:
            stretched.append(p)
            if is_vowel(p):
                for _ in range(repetition_count - 1):
                    stretched.append(p)

        return self.detect(word_phonemes, stretched)

    def find_melisma_opportunities(
        self,
        word: str,
        word_phonemes: list[str]
    ) -> list[dict]:
        """Find potential melisma opportunities for a word.

        Suggests how stretching different syllables could create wordplay.

        Args:
            word: Original word
            word_phonemes: Original phonemes

        Returns:
            List of melisma opportunities
        """
        from src.detection.phonetic_detectors import is_vowel

        opportunities = []

        # Find vowel positions
        vowel_indices = [i for i, p in enumerate(word_phonemes) if is_vowel(p)]

        # Try stretching each vowel
        for idx in vowel_indices:
            for stretch in [2, 3]:
                stretched = word_phonemes.copy()
                # Insert repeated vowel
                for _ in range(stretch - 1):
                    stretched.insert(idx + 1, word_phonemes[idx])

                stretched_key = " ".join(stretched)
                if stretched_key in self.reverse_index:
                    matches = self.reverse_index[stretched_key]
                    if matches:
                        opportunities.append({
                            "vowel_index": idx,
                            "vowel": word_phonemes[idx],
                            "stretch_factor": stretch,
                            "reveals": matches[:5],
                        })

        return opportunities


# =============================================================================
# 21. SAMPLE FLIP DETECTOR
# =============================================================================

class SampleFlipDetector:
    """Detect sample flips - phonemes from sampled lyrics in new context.

    A sample flip is when an artist uses a phonetic pattern from
    a classic song in a new context, creating a clever reference.
    """

    def __init__(
        self,
        sample_database: dict[str, list[str]],
        similarity_threshold: float = 0.85
    ):
        """
        Args:
            sample_database: song_id → phoneme sequence mapping
            similarity_threshold: Minimum similarity for match
        """
        self.samples = sample_database
        self.similarity_threshold = similarity_threshold

    def phoneme_similarity(self, p1: list[str], p2: list[str]) -> float:
        """Calculate Levenshtein similarity between phoneme sequences."""
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
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,
                        dp[i][j - 1] + 1,
                        dp[i - 1][j - 1] + 1
                    )

        distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len)

    def detect(
        self,
        new_bar_phonemes: list[str]
    ) -> list[SampleFlipMatch]:
        """Detect if bar contains flipped samples.

        Args:
            new_bar_phonemes: Phoneme sequence of new bar

        Returns:
            List of detected sample flips
        """
        results = []

        for song_id, sample_phonemes in self.samples.items():
            similarity = self.phoneme_similarity(new_bar_phonemes, sample_phonemes)

            if similarity >= self.similarity_threshold:
                results.append(SampleFlipMatch(
                    original_song=song_id,
                    original_phonemes=sample_phonemes,
                    new_phonemes=new_bar_phonemes,
                    similarity=similarity,
                    type="sample_flip",
                    confidence=similarity
                ))

        # Also check subsequences for partial flips
        partial_results = self._detect_partial_flips(new_bar_phonemes)
        results.extend(partial_results)

        return sorted(results, key=lambda x: -x.confidence)

    def _detect_partial_flips(
        self,
        new_bar_phonemes: list[str],
        min_window: int = 5,
        max_window: int = 15
    ) -> list[SampleFlipMatch]:
        """Detect partial sample flips using sliding window.

        Args:
            new_bar_phonemes: New bar phonemes
            min_window: Minimum window size
            max_window: Maximum window size

        Returns:
            List of partial flip matches
        """
        results = []

        for song_id, sample_phonemes in self.samples.items():
            for window_size in range(min_window, min(len(sample_phonemes), max_window)):
                for start in range(len(sample_phonemes) - window_size + 1):
                    window = sample_phonemes[start:start + window_size]

                    if self._is_subsequence(window, new_bar_phonemes):
                        results.append(SampleFlipMatch(
                            original_song=song_id,
                            original_phonemes=window,
                            new_phonemes=new_bar_phonemes,
                            similarity=0.7,
                            type="partial_sample_flip",
                            confidence=0.7
                        ))
                        break  # Found one match for this window size

        return results

    def _is_subsequence(
        self,
        pattern: list[str],
        sequence: list[str]
    ) -> bool:
        """Check if pattern appears as contiguous subsequence.

        Args:
            pattern: Pattern to find
            sequence: Sequence to search in

        Returns:
            True if pattern found
        """
        pattern_str = " ".join(strip_stress(p) for p in pattern)
        sequence_str = " ".join(strip_stress(p) for p in sequence)
        return pattern_str in sequence_str

    def add_sample(self, song_id: str, phonemes: list[str]) -> None:
        """Add a sample to the database.

        Args:
            song_id: Identifier for the song
            phonemes: Phoneme sequence from the sample
        """
        self.samples[song_id] = phonemes

    def find_similar_samples(
        self,
        query_phonemes: list[str],
        top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Find samples most similar to query.

        Args:
            query_phonemes: Query phoneme sequence
            top_k: Number of results to return

        Returns:
            List of (song_id, similarity) tuples
        """
        similarities = []

        for song_id, sample_phonemes in self.samples.items():
            sim = self.phoneme_similarity(query_phonemes, sample_phonemes)
            similarities.append((song_id, sim))

        return sorted(similarities, key=lambda x: -x[1])[:top_k]


# =============================================================================
# CLASSIC SAMPLE DATABASE
# =============================================================================

# Pre-built database of classic samples frequently flipped
CLASSIC_SAMPLES = {
    # Soul/R&B samples
    "james_brown_funky_drummer": ["F AH1 NG K IY0", "D R AH1 M ER0"],
    "marvin_gaye_whats_going_on": ["W AH1 T S", "G OW1 IH0 NG", "AA1 N"],
    "isaac_hayes_walk_on_by": ["W AO1 K", "AA1 N", "B AY1"],

    # Hip-hop classics
    "grandmaster_flash_message": ["D OW1 N T", "P UH1 SH", "M IY1"],
    "notorious_big_juicy": ["IH1 T", "W AA1 Z", "AO1 L", "AH0", "D R IY1 M"],
    "nas_world_is_yours": ["DH AH0", "W ER1 L D", "IH1 Z", "Y AO1 R Z"],

    # Rock samples
    "led_zeppelin_when_levee_breaks": ["W EH1 N", "DH AH0", "L EH1 V IY0", "B R EY1 K S"],
    "queen_under_pressure": ["P R EH1 SH ER0"],
}


def get_sample_database() -> dict[str, list[str]]:
    """Get the default sample database.

    Returns:
        Sample database dictionary
    """
    # Convert string phonemes to lists
    return {
        song_id: phonemes if isinstance(phonemes, list) else phonemes.split()
        for song_id, phonemes in CLASSIC_SAMPLES.items()
    }
