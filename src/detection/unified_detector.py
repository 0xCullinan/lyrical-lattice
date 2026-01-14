"""
File: src/detection/unified_detector.py
Purpose: Unified orchestrator for all 21 wordplay device detectors
"""

import time
from typing import Optional
from dataclasses import dataclass

from src.detection.models import (
    DeviceType,
    WordplayResult,
    DetectionResult,
)
from src.detection.phonetic_detectors import (
    HomophoneDetector,
    OronymDetector,
    PerfectRhymeDetector,
    SlantRhymeDetector,
    AssonanceDetector,
    ConsonanceDetector,
    AlliterationDetector,
    InternalRhymeDetector,
    MultisyllabicRhymeDetector,
    CompoundRhymeDetector,
    OnomatopoeiaDetector,
    EuphonyCacophonyDetector,
    StackedRhymeDetector,
)
from src.detection.semantic_detectors import (
    PunDetector,
    DoubleEntendreDetector,
    MalapropismDetector,
    MondegreenDetector,
)
from src.detection.music_detectors import (
    PolyrhythmicRhymeDetector,
    BreathRhymeDetector,
    MelismaDetector,
    SampleFlipDetector,
    get_sample_database,
)


@dataclass
class DetectorConfig:
    """Configuration for the unified detector."""
    # Which detector categories to enable
    enable_phonetic: bool = True
    enable_semantic: bool = True
    enable_music: bool = True

    # Specific detectors to enable (None = all)
    enabled_devices: Optional[set[DeviceType]] = None

    # Thresholds
    min_confidence: float = 0.5
    max_results_per_device: int = 10

    # Music-specific settings
    bpm: int = 90


class UnifiedWordplayDetector:
    """Orchestrates all 21 wordplay device detectors.

    This class provides a unified interface for detecting all types of
    wordplay devices from phoneme sequences.
    """

    def __init__(
        self,
        reverse_index: dict[str, list[str]],
        word_frequencies: Optional[dict[str, int]] = None,
        word_embeddings: Optional[dict] = None,
        sample_database: Optional[dict[str, list[str]]] = None,
        config: Optional[DetectorConfig] = None,
    ):
        """
        Args:
            reverse_index: ARPAbet phoneme → words mapping
            word_frequencies: Optional word frequency dict
            word_embeddings: Optional word embeddings for semantic analysis
            sample_database: Optional sample database for sample flip detection
            config: Detector configuration
        """
        self.reverse_index = reverse_index
        self.word_frequencies = word_frequencies or {}
        self.word_embeddings = word_embeddings or {}
        self.config = config or DetectorConfig()

        # Initialize all detectors
        self._init_phonetic_detectors()
        self._init_semantic_detectors()
        self._init_music_detectors(sample_database)

    def _init_phonetic_detectors(self) -> None:
        """Initialize all 13 phonetic detectors."""
        # 1. Homophone
        self.homophone_detector = HomophoneDetector(self.reverse_index)

        # 2. Oronym
        self.oronym_detector = OronymDetector(
            self.reverse_index,
            self.word_frequencies
        )

        # 3. Perfect Rhyme
        self.perfect_rhyme_detector = PerfectRhymeDetector(self.reverse_index)

        # 4. Slant Rhyme
        self.slant_rhyme_detector = SlantRhymeDetector(
            self.perfect_rhyme_detector.rhyme_index
        )

        # 5. Assonance
        self.assonance_detector = AssonanceDetector()

        # 6. Consonance
        self.consonance_detector = ConsonanceDetector()

        # 7. Alliteration
        self.alliteration_detector = AlliterationDetector()

        # 8. Internal Rhyme
        self.internal_rhyme_detector = InternalRhymeDetector(
            self.perfect_rhyme_detector
        )

        # 9. Multisyllabic Rhyme
        self.multisyllabic_detector = MultisyllabicRhymeDetector(
            self.reverse_index
        )

        # 10. Compound Rhyme
        self.compound_rhyme_detector = CompoundRhymeDetector(
            self.perfect_rhyme_detector
        )

        # 11. Onomatopoeia
        self.onomatopoeia_detector = OnomatopoeiaDetector()

        # 12. Euphony/Cacophony
        self.euphony_detector = EuphonyCacophonyDetector()

        # 13. Stacked Rhyme
        self.stacked_rhyme_detector = StackedRhymeDetector(
            self.perfect_rhyme_detector
        )

    def _init_semantic_detectors(self) -> None:
        """Initialize all 4 semantic detectors."""
        # 14. Pun
        self.pun_detector = PunDetector(
            self.homophone_detector,
            self.word_embeddings
        )

        # 15. Double Entendre
        self.double_entendre_detector = DoubleEntendreDetector()

        # 16. Malapropism
        self.malapropism_detector = MalapropismDetector(self.reverse_index)

        # 17. Mondegreen
        self.mondegreen_detector = MondegreenDetector(
            self.oronym_detector,
            self.word_frequencies
        )

    def _init_music_detectors(
        self,
        sample_database: Optional[dict[str, list[str]]]
    ) -> None:
        """Initialize all 4 music-specific detectors."""
        # 18. Polyrhythmic Rhyme
        self.polyrhythmic_detector = PolyrhythmicRhymeDetector(
            self.perfect_rhyme_detector,
            self.config.bpm
        )

        # 19. Breath Rhyme
        self.breath_rhyme_detector = BreathRhymeDetector(
            self.perfect_rhyme_detector
        )

        # 20. Melisma
        self.melisma_detector = MelismaDetector(self.reverse_index)

        # 21. Sample Flip
        samples = sample_database or get_sample_database()
        self.sample_flip_detector = SampleFlipDetector(samples)

    def _is_enabled(self, device_type: DeviceType) -> bool:
        """Check if a device type is enabled."""
        if self.config.enabled_devices is not None:
            return device_type in self.config.enabled_devices
        return True

    def detect_all(
        self,
        bar_phonemes: list[list[str]],
        bar_words: Optional[list[str]] = None,
        bar_timing: Optional[list[tuple[list[str], float]]] = None,
    ) -> DetectionResult:
        """Run all enabled detectors on a bar.

        Args:
            bar_phonemes: List of phoneme sequences, one per word
            bar_words: Optional list of word strings
            bar_timing: Optional list of (phonemes, timestamp) for music detection

        Returns:
            DetectionResult with all matches
        """
        start_time = time.perf_counter()
        results: list[WordplayResult] = []

        # Flatten phonemes for some detectors
        flat_phonemes = [p for word_phones in bar_phonemes for p in word_phones]

        # Run phonetic detectors
        if self.config.enable_phonetic:
            results.extend(self._run_phonetic_detectors(
                bar_phonemes, flat_phonemes, bar_words
            ))

        # Run semantic detectors
        if self.config.enable_semantic and bar_words:
            results.extend(self._run_semantic_detectors(
                bar_phonemes, bar_words
            ))

        # Run music detectors
        if self.config.enable_music and bar_timing:
            results.extend(self._run_music_detectors(
                bar_phonemes, flat_phonemes, bar_timing
            ))

        # Filter by confidence
        results = [
            r for r in results
            if r.confidence >= self.config.min_confidence
        ]

        # Sort by confidence
        results.sort(key=lambda x: -x.confidence)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        return DetectionResult(
            input_text=" ".join(bar_words) if bar_words else None,
            input_phonemes=flat_phonemes,
            wordplay_matches=results,
            processing_time_ms=elapsed_ms,
        )

    def _run_phonetic_detectors(
        self,
        bar_phonemes: list[list[str]],
        flat_phonemes: list[str],
        bar_words: Optional[list[str]],
    ) -> list[WordplayResult]:
        """Run all phonetic detectors."""
        results = []

        # 1. Homophones
        if self._is_enabled(DeviceType.HOMOPHONE):
            matches = self.homophone_detector.detect_in_bar(bar_phonemes)
            for m in matches[:self.config.max_results_per_device]:
                results.append(WordplayResult(
                    device_type=DeviceType.HOMOPHONE,
                    match=m,
                    confidence=m.confidence,
                ))

        # 2. Oronyms
        if self._is_enabled(DeviceType.ORONYM):
            matches = self.oronym_detector.detect(flat_phonemes)
            for m in matches[:self.config.max_results_per_device]:
                results.append(WordplayResult(
                    device_type=DeviceType.ORONYM,
                    match=m,
                    confidence=m.confidence,
                ))

        # 3. Perfect Rhyme
        if self._is_enabled(DeviceType.PERFECT_RHYME):
            for i, word_phones in enumerate(bar_phonemes):
                matches = self.perfect_rhyme_detector.detect(word_phones)
                for m in matches:
                    results.append(WordplayResult(
                        device_type=DeviceType.PERFECT_RHYME,
                        match=m,
                        confidence=m.confidence,
                        word_indices=[i],
                    ))

        # 4. Slant Rhyme
        if self._is_enabled(DeviceType.SLANT_RHYME):
            for i, word_phones in enumerate(bar_phonemes):
                matches = self.slant_rhyme_detector.detect(word_phones)
                for m in matches[:3]:  # Top 3 per word
                    results.append(WordplayResult(
                        device_type=DeviceType.SLANT_RHYME,
                        match=m,
                        confidence=m.confidence,
                        word_indices=[i],
                    ))

        # 5. Assonance
        if self._is_enabled(DeviceType.ASSONANCE):
            matches = self.assonance_detector.detect(bar_phonemes)
            for m in matches[:self.config.max_results_per_device]:
                results.append(WordplayResult(
                    device_type=DeviceType.ASSONANCE,
                    match=m,
                    confidence=m.confidence,
                    word_indices=m.word_indices,
                ))

        # 6. Consonance
        if self._is_enabled(DeviceType.CONSONANCE):
            matches = self.consonance_detector.detect(bar_phonemes)
            for m in matches[:self.config.max_results_per_device]:
                results.append(WordplayResult(
                    device_type=DeviceType.CONSONANCE,
                    match=m,
                    confidence=m.confidence,
                    word_indices=m.word_indices,
                ))

        # 7. Alliteration
        if self._is_enabled(DeviceType.ALLITERATION):
            matches = self.alliteration_detector.detect(bar_phonemes)
            for m in matches:
                results.append(WordplayResult(
                    device_type=DeviceType.ALLITERATION,
                    match=m,
                    confidence=m.confidence,
                    word_indices=m.word_indices,
                ))

        # 8. Internal Rhyme
        if self._is_enabled(DeviceType.INTERNAL_RHYME):
            matches = self.internal_rhyme_detector.detect(bar_phonemes)
            for m in matches:
                results.append(WordplayResult(
                    device_type=DeviceType.INTERNAL_RHYME,
                    match=m,
                    confidence=m.confidence,
                    word_indices=list(m.word_indices),
                ))

        # 9. Multisyllabic Rhyme
        if self._is_enabled(DeviceType.MULTISYLLABIC_RHYME):
            for i, word_phones in enumerate(bar_phonemes):
                matches = self.multisyllabic_detector.detect(word_phones)
                for m in matches:
                    results.append(WordplayResult(
                        device_type=DeviceType.MULTISYLLABIC_RHYME,
                        match=m,
                        confidence=m.confidence,
                        word_indices=[i],
                    ))

        # 10. Compound Rhyme (check consecutive word pairs)
        if self._is_enabled(DeviceType.COMPOUND_RHYME):
            for i in range(len(bar_phonemes) - 1):
                multi_word = bar_phonemes[i:i + 2]
                matches = self.compound_rhyme_detector.detect(multi_word)
                for m in matches:
                    results.append(WordplayResult(
                        device_type=DeviceType.COMPOUND_RHYME,
                        match=m,
                        confidence=m.confidence,
                        word_indices=[i, i + 1],
                    ))

        # 11. Onomatopoeia
        if self._is_enabled(DeviceType.ONOMATOPOEIA) and bar_words:
            for i, (word, phones) in enumerate(zip(bar_words, bar_phonemes)):
                match = self.onomatopoeia_detector.detect(word, phones)
                if match:
                    results.append(WordplayResult(
                        device_type=DeviceType.ONOMATOPOEIA,
                        match=match,
                        confidence=match.confidence,
                        word_indices=[i],
                    ))

        # 12. Euphony/Cacophony
        if self._is_enabled(DeviceType.EUPHONY) or self._is_enabled(DeviceType.CACOPHONY):
            match = self.euphony_detector.detect(flat_phonemes)
            device_type = DeviceType.EUPHONY if match.type == "euphony" else DeviceType.CACOPHONY
            if self._is_enabled(device_type):
                results.append(WordplayResult(
                    device_type=device_type,
                    match=match,
                    confidence=match.confidence,
                ))

        # 13. Stacked Rhyme
        if self._is_enabled(DeviceType.STACKED_RHYME):
            match = self.stacked_rhyme_detector.detect(bar_phonemes)
            if match:
                results.append(WordplayResult(
                    device_type=DeviceType.STACKED_RHYME,
                    match=match,
                    confidence=match.confidence,
                ))

        return results

    def _run_semantic_detectors(
        self,
        bar_phonemes: list[list[str]],
        bar_words: list[str],
    ) -> list[WordplayResult]:
        """Run all semantic detectors."""
        results = []

        # 14. Pun
        if self._is_enabled(DeviceType.PUN):
            for i, (word, phones) in enumerate(zip(bar_words, bar_phonemes)):
                context = bar_words[:i] + bar_words[i + 1:]
                if self.word_embeddings:
                    match = self.pun_detector.detect(word, phones, context)
                else:
                    match = self.pun_detector.detect_without_embeddings(
                        word, phones, context
                    )
                if match:
                    results.append(WordplayResult(
                        device_type=DeviceType.PUN,
                        match=match,
                        confidence=match.confidence,
                        word_indices=[i],
                    ))

        # 15. Double Entendre
        if self._is_enabled(DeviceType.DOUBLE_ENTENDRE):
            matches = self.double_entendre_detector.detect_phrase(bar_words)
            for m in matches:
                word_idx = bar_words.index(m.word) if m.word in bar_words else -1
                results.append(WordplayResult(
                    device_type=DeviceType.DOUBLE_ENTENDRE,
                    match=m,
                    confidence=m.confidence,
                    word_indices=[word_idx] if word_idx >= 0 else [],
                ))

        # 16. Malapropism
        if self._is_enabled(DeviceType.MALAPROPISM):
            full_sentence = " ".join(bar_words)
            for i, (word, phones) in enumerate(zip(bar_words, bar_phonemes)):
                match = self.malapropism_detector.detect(word, phones, full_sentence)
                if match:
                    results.append(WordplayResult(
                        device_type=DeviceType.MALAPROPISM,
                        match=match,
                        confidence=match.confidence,
                        word_indices=[i],
                    ))

        # 17. Mondegreen
        if self._is_enabled(DeviceType.MONDEGREEN):
            flat_phonemes = [p for word_phones in bar_phonemes for p in word_phones]
            matches = self.mondegreen_detector.detect(flat_phonemes, bar_words)
            for m in matches[:self.config.max_results_per_device]:
                results.append(WordplayResult(
                    device_type=DeviceType.MONDEGREEN,
                    match=m,
                    confidence=m.plausibility,
                ))

        return results

    def _run_music_detectors(
        self,
        bar_phonemes: list[list[str]],
        flat_phonemes: list[str],
        bar_timing: list[tuple[list[str], float]],
    ) -> list[WordplayResult]:
        """Run all music-specific detectors."""
        results = []

        # 18. Polyrhythmic Rhyme
        if self._is_enabled(DeviceType.POLYRHYTHMIC_RHYME):
            # Need multiple bars for polyrhythmic detection
            # For single bar, create artificial bars
            bars = [bar_timing]
            match = self.polyrhythmic_detector.detect(bars)
            if match:
                results.append(WordplayResult(
                    device_type=DeviceType.POLYRHYTHMIC_RHYME,
                    match=match,
                    confidence=match.confidence,
                ))

        # 19. Breath Rhyme
        if self._is_enabled(DeviceType.BREATH_RHYME):
            matches = self.breath_rhyme_detector.detect(
                bar_timing, self.config.bpm
            )
            for m in matches:
                results.append(WordplayResult(
                    device_type=DeviceType.BREATH_RHYME,
                    match=m,
                    confidence=0.8,
                    timestamp=m.beat_position,
                ))

        # 20. Melisma (simplified - would need audio analysis for real detection)
        if self._is_enabled(DeviceType.MELISMA_WORDPLAY):
            for i, word_phones in enumerate(bar_phonemes):
                # Check for melisma opportunities
                opportunities = self.melisma_detector.find_melisma_opportunities(
                    "", word_phones
                )
                for opp in opportunities[:2]:
                    results.append(WordplayResult(
                        device_type=DeviceType.MELISMA_WORDPLAY,
                        match=opp,
                        confidence=0.6,
                        word_indices=[i],
                    ))

        # 21. Sample Flip
        if self._is_enabled(DeviceType.SAMPLE_FLIP):
            matches = self.sample_flip_detector.detect(flat_phonemes)
            for m in matches[:self.config.max_results_per_device]:
                results.append(WordplayResult(
                    device_type=DeviceType.SAMPLE_FLIP,
                    match=m,
                    confidence=m.confidence,
                ))

        return results

    def detect_single_device(
        self,
        device_type: DeviceType,
        bar_phonemes: list[list[str]],
        bar_words: Optional[list[str]] = None,
        bar_timing: Optional[list[tuple[list[str], float]]] = None,
    ) -> list[WordplayResult]:
        """Run a single specific detector.

        Args:
            device_type: Which device to detect
            bar_phonemes: Phoneme sequences per word
            bar_words: Optional word strings
            bar_timing: Optional timing information

        Returns:
            List of matches for that device type
        """
        # Temporarily modify config to only enable this device
        original_devices = self.config.enabled_devices
        self.config.enabled_devices = {device_type}

        try:
            result = self.detect_all(bar_phonemes, bar_words, bar_timing)
            return result.wordplay_matches
        finally:
            self.config.enabled_devices = original_devices

    def get_rhyme_suggestions(
        self,
        phonemes: list[str],
        include_slant: bool = True,
        include_multisyllabic: bool = True,
        max_results: int = 20,
    ) -> dict[str, list[str]]:
        """Get rhyme suggestions for a word.

        Args:
            phonemes: ARPAbet phoneme sequence
            include_slant: Include slant rhymes
            include_multisyllabic: Include multisyllabic rhymes
            max_results: Maximum results per category

        Returns:
            Dictionary of rhyme categories to word lists
        """
        suggestions = {
            "perfect": [],
            "slant": [],
            "multisyllabic": [],
        }

        # Perfect rhymes
        perfect_matches = self.perfect_rhyme_detector.detect(phonemes)
        if perfect_matches:
            suggestions["perfect"] = perfect_matches[0].matches[:max_results]

        # Slant rhymes
        if include_slant:
            slant_matches = self.slant_rhyme_detector.detect(phonemes, max_results)
            for m in slant_matches:
                suggestions["slant"].extend(m.matches[:5])
            suggestions["slant"] = suggestions["slant"][:max_results]

        # Multisyllabic rhymes
        if include_multisyllabic:
            multi_matches = self.multisyllabic_detector.detect(phonemes)
            for m in multi_matches:
                suggestions["multisyllabic"].extend(m.matches[:5])
            suggestions["multisyllabic"] = suggestions["multisyllabic"][:max_results]

        return suggestions

    def analyze_bar(
        self,
        bar_phonemes: list[list[str]],
        bar_words: Optional[list[str]] = None,
    ) -> dict:
        """Get comprehensive analysis of a bar.

        Args:
            bar_phonemes: Phoneme sequences per word
            bar_words: Optional word strings

        Returns:
            Analysis dictionary with metrics
        """
        flat_phonemes = [p for wp in bar_phonemes for p in wp]

        analysis = {
            "word_count": len(bar_phonemes),
            "phoneme_count": len(flat_phonemes),
            "sound_texture": None,
            "rhyme_density": 0.0,
            "device_count": {},
        }

        # Sound texture analysis
        texture = self.euphony_detector.detect(flat_phonemes)
        analysis["sound_texture"] = {
            "type": texture.type,
            "euphony_score": texture.euphony_score,
            "cacophony_score": texture.cacophony_score,
        }

        # Rhyme density
        stacked = self.stacked_rhyme_detector.detect(bar_phonemes)
        if stacked:
            analysis["rhyme_density"] = stacked.density

        # Count devices found
        result = self.detect_all(bar_phonemes, bar_words)
        for match in result.wordplay_matches:
            device = match.device_type.value
            analysis["device_count"][device] = (
                analysis["device_count"].get(device, 0) + 1
            )

        return analysis
