"""
File: src/detection/semantic_detectors.py
Purpose: 4 phonetic+semantic wordplay device detectors
"""

import math
from typing import Optional, Callable

import numpy as np

from src.detection.models import (
    PunMatch,
    DoubleEntendreMatch,
    MalapropismMatch,
    MondegreenMatch,
)
from src.detection.phonetic_detectors import (
    HomophoneDetector,
    OronymDetector,
    strip_stress,
)


# =============================================================================
# 14. PUN DETECTOR
# =============================================================================

class PunDetector:
    """Detect puns - homophones where BOTH meanings fit the context.

    A pun requires:
    1. A homophone relationship (same sound, different words)
    2. Both meanings making semantic sense in context
    """

    def __init__(
        self,
        homophone_detector: HomophoneDetector,
        word_embeddings: Optional[dict[str, np.ndarray]] = None,
        context_window: int = 5
    ):
        """
        Args:
            homophone_detector: For finding homophones
            word_embeddings: Optional word → embedding vector dict
            context_window: Words to consider for context
        """
        self.homophone_detector = homophone_detector
        self.embeddings = word_embeddings or {}
        self.context_window = context_window

    def semantic_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between word embeddings.

        Args:
            word1, word2: Words to compare

        Returns:
            Similarity score (0-1)
        """
        w1 = word1.lower()
        w2 = word2.lower()

        if w1 not in self.embeddings or w2 not in self.embeddings:
            return 0.0

        v1 = self.embeddings[w1]
        v2 = self.embeddings[w2]

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot / (norm1 * norm2))

    def context_fit(self, word: str, context_words: list[str]) -> float:
        """Calculate how well a word fits the context.

        Args:
            word: Word to evaluate
            context_words: Surrounding words for context

        Returns:
            Average similarity to context (0-1)
        """
        if not context_words:
            return 0.5

        similarities = [
            self.semantic_similarity(word, ctx_word)
            for ctx_word in context_words
            if ctx_word.lower() != word.lower()
        ]

        return float(np.mean(similarities)) if similarities else 0.0

    def detect(
        self,
        word: str,
        word_phonemes: list[str],
        context_words: list[str]
    ) -> Optional[PunMatch]:
        """Detect if word is used as a pun.

        Args:
            word: The spoken word
            word_phonemes: ARPAbet phoneme sequence
            context_words: Surrounding words for context

        Returns:
            PunMatch if pun detected, None otherwise
        """
        # Step 1: Find homophones
        homophones = self.homophone_detector.detect(word_phonemes)
        if not homophones:
            return None

        homophone_words = homophones[0].words

        if len(homophone_words) < 2:
            return None

        # Step 2: Check semantic fit for each homophone
        fits = {}
        for hw in homophone_words:
            fits[hw] = self.context_fit(hw, context_words)

        # Step 3: Pun requires 2+ meanings to fit context
        fitting_words = [w for w, fit in fits.items() if fit > 0.3]

        if len(fitting_words) >= 2:
            return PunMatch(
                spoken_word=word,
                homophones=homophone_words,
                context_fits=fits,
                pun_strength=min(fits.values()),
                confidence=len(fitting_words) / len(homophone_words),
                type="pun"
            )

        return None

    def detect_without_embeddings(
        self,
        word: str,
        word_phonemes: list[str],
        context_words: list[str]
    ) -> Optional[PunMatch]:
        """Detect potential puns without semantic embeddings.

        Uses simple heuristics instead of embeddings.
        """
        homophones = self.homophone_detector.detect(word_phonemes)
        if not homophones:
            return None

        homophone_words = homophones[0].words

        if len(homophone_words) < 2:
            return None

        # Simple heuristic: check if any context word shares prefix/suffix
        context_lower = [w.lower() for w in context_words]

        fits = {}
        for hw in homophone_words:
            hw_lower = hw.lower()
            # Check for shared prefix (3+ chars) or suffix (3+ chars)
            score = 0.0
            for ctx in context_lower:
                if len(hw_lower) >= 3 and len(ctx) >= 3:
                    if hw_lower[:3] == ctx[:3] or hw_lower[-3:] == ctx[-3:]:
                        score += 0.3
                # Check if words are related (same stem)
                if hw_lower in ctx or ctx in hw_lower:
                    score += 0.5
            fits[hw] = min(score, 1.0)

        fitting_words = [w for w, fit in fits.items() if fit > 0.2]

        if len(fitting_words) >= 2:
            return PunMatch(
                spoken_word=word,
                homophones=homophone_words,
                context_fits=fits,
                pun_strength=min(fits.values()) if fits.values() else 0.0,
                confidence=len(fitting_words) / len(homophone_words),
                type="pun"
            )

        return None


# =============================================================================
# 15. DOUBLE ENTENDRE DETECTOR
# =============================================================================

class DoubleEntendreDetector:
    """Detect double entendres - innocent surface with hidden meaning.

    Uses a lexicon of words with known double meanings.
    """

    # Pre-built entendre lexicon: word → {surface_meaning, hidden_meaning, domain}
    ENTENDRE_WORDS = {
        # Romance/intimacy domain
        "sugar": {"surface": "sweetener", "hidden": "affection/intimacy", "domain": "romance"},
        "cherry": {"surface": "fruit", "hidden": "innocence/virginity", "domain": "romance"},
        "honey": {"surface": "sweetener", "hidden": "term of endearment", "domain": "romance"},
        "peach": {"surface": "fruit", "hidden": "attractive person", "domain": "romance"},
        "juicy": {"surface": "moist/wet", "hidden": "provocative/scandalous", "domain": "romance"},
        "hot": {"surface": "high temperature", "hidden": "attractive/arousing", "domain": "romance"},
        "wet": {"surface": "containing water", "hidden": "aroused", "domain": "romance"},
        "come": {"surface": "arrive", "hidden": "orgasm", "domain": "romance"},
        "ride": {"surface": "travel on vehicle", "hidden": "sexual activity", "domain": "romance"},

        # Varied domains
        "rock": {"surface": "stone", "hidden": "drugs/strength/jewelry", "domain": "varied"},
        "blow": {"surface": "wind/breath", "hidden": "cocaine/oral sex", "domain": "varied"},
        "hard": {"surface": "difficult/solid", "hidden": "erection/tough", "domain": "varied"},
        "high": {"surface": "elevated", "hidden": "intoxicated", "domain": "varied"},
        "loaded": {"surface": "full/carrying", "hidden": "wealthy/intoxicated", "domain": "varied"},
        "stoned": {"surface": "pelted with rocks", "hidden": "intoxicated", "domain": "varied"},
        "baked": {"surface": "cooked", "hidden": "intoxicated", "domain": "varied"},
        "lit": {"surface": "illuminated", "hidden": "intoxicated/exciting", "domain": "varied"},
        "fire": {"surface": "flames", "hidden": "excellent/dismiss", "domain": "varied"},
        "ice": {"surface": "frozen water", "hidden": "diamonds/meth/coldness", "domain": "varied"},
        "snow": {"surface": "precipitation", "hidden": "cocaine", "domain": "varied"},
        "pipe": {"surface": "tube", "hidden": "smoking device", "domain": "varied"},
        "joint": {"surface": "connection point", "hidden": "marijuana cigarette", "domain": "varied"},
        "grass": {"surface": "lawn plants", "hidden": "marijuana", "domain": "varied"},
        "pot": {"surface": "container", "hidden": "marijuana", "domain": "varied"},
        "weed": {"surface": "unwanted plant", "hidden": "marijuana", "domain": "varied"},
        "crack": {"surface": "fracture", "hidden": "cocaine/joke", "domain": "varied"},
        "dope": {"surface": "stupid", "hidden": "drugs/excellent", "domain": "varied"},
        "smack": {"surface": "hit/kiss", "hidden": "heroin", "domain": "varied"},

        # Music/street domain
        "bars": {"surface": "rods/pubs", "hidden": "rap lyrics/prison", "domain": "music"},
        "flow": {"surface": "movement of liquid", "hidden": "rap delivery style", "domain": "music"},
        "heat": {"surface": "warmth", "hidden": "firearms/pressure", "domain": "music"},
        "piece": {"surface": "portion", "hidden": "firearm", "domain": "music"},
        "strap": {"surface": "band/strip", "hidden": "firearm", "domain": "music"},
        "tool": {"surface": "implement", "hidden": "firearm", "domain": "music"},
        "iron": {"surface": "metal/appliance", "hidden": "firearm", "domain": "music"},
        "beef": {"surface": "meat", "hidden": "conflict/dispute", "domain": "music"},
        "green": {"surface": "color", "hidden": "money/marijuana", "domain": "music"},
        "bread": {"surface": "food", "hidden": "money", "domain": "music"},
        "paper": {"surface": "material", "hidden": "money", "domain": "music"},
        "cake": {"surface": "dessert", "hidden": "money/drugs", "domain": "music"},
        "cheese": {"surface": "dairy product", "hidden": "money", "domain": "music"},
        "bands": {"surface": "music groups", "hidden": "money (stacks)", "domain": "music"},
        "stacks": {"surface": "piles", "hidden": "money", "domain": "music"},
        "whip": {"surface": "lash", "hidden": "car", "domain": "music"},
        "drip": {"surface": "liquid falling", "hidden": "style/fashion", "domain": "music"},
        "sauce": {"surface": "condiment", "hidden": "style/swagger", "domain": "music"},
        "fresh": {"surface": "new/clean", "hidden": "stylish", "domain": "music"},
        "cold": {"surface": "low temperature", "hidden": "ruthless/stylish", "domain": "music"},
        "sick": {"surface": "ill", "hidden": "excellent", "domain": "music"},
        "dirty": {"surface": "unclean", "hidden": "raw/authentic/illegal", "domain": "music"},
    }

    # Domain-specific context keywords
    DOMAIN_KEYWORDS = {
        "romance": {"love", "baby", "girl", "boy", "heart", "night", "bed", "body",
                    "touch", "kiss", "feel", "want", "need", "desire", "sweet"},
        "varied": {"party", "night", "smoke", "drink", "feel", "high", "trip",
                   "roll", "chill", "vibe", "zone"},
        "music": {"rap", "flow", "bars", "beat", "mic", "rhyme", "verse", "hook",
                  "track", "studio", "record", "street", "hood", "gang", "crew",
                  "money", "cash", "rich", "broke", "hustle", "grind"},
    }

    def __init__(
        self,
        custom_lexicon: Optional[dict] = None
    ):
        """
        Args:
            custom_lexicon: Optional additional entendre words
        """
        self.lexicon = {**self.ENTENDRE_WORDS}
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)

    def get_domain_keywords(self, domain: str) -> set[str]:
        """Get keywords associated with a domain."""
        return self.DOMAIN_KEYWORDS.get(domain, set())

    def detect_single_word(
        self,
        word: str,
        context: list[str]
    ) -> Optional[DoubleEntendreMatch]:
        """Check if word has double meaning in context.

        Args:
            word: Word to check
            context: Surrounding words

        Returns:
            DoubleEntendreMatch if detected
        """
        word_lower = word.lower()

        if word_lower not in self.lexicon:
            return None

        entry = self.lexicon[word_lower]

        # Check if context suggests hidden meaning
        domain_keywords = self.get_domain_keywords(entry["domain"])
        context_lower = {w.lower() for w in context}
        context_overlap = len(context_lower & domain_keywords)

        if context_overlap > 0:
            return DoubleEntendreMatch(
                word=word,
                surface_meaning=entry["surface"],
                hidden_meaning=entry["hidden"],
                domain=entry["domain"],
                context_support=context_overlap,
                confidence=min(context_overlap / 3.0, 0.9),
                type="double_entendre"
            )

        return None

    def detect_phrase(self, phrase_words: list[str]) -> list[DoubleEntendreMatch]:
        """Scan phrase for double entendres.

        Args:
            phrase_words: List of words in phrase

        Returns:
            List of detected double entendres
        """
        results = []

        for i, word in enumerate(phrase_words):
            context = phrase_words[:i] + phrase_words[i + 1:]
            match = self.detect_single_word(word, context)
            if match:
                results.append(match)

        return results


# =============================================================================
# 16. MALAPROPISM DETECTOR
# =============================================================================

class MalapropismDetector:
    """Detect malapropisms - wrong word that sounds like the right one.

    A malapropism occurs when a phonetically similar word is used
    incorrectly, making the sentence less coherent.
    """

    def __init__(
        self,
        reverse_index: dict[str, list[str]],
        perplexity_fn: Optional[Callable[[str], float]] = None
    ):
        """
        Args:
            reverse_index: Phoneme → words mapping
            perplexity_fn: Optional function to calculate sentence perplexity
        """
        self.reverse_index = reverse_index
        self.perplexity_fn = perplexity_fn

    def phoneme_similarity(self, p1: list[str], p2: list[str]) -> float:
        """Calculate phoneme sequence similarity using Levenshtein."""
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

    def find_phonetically_similar(
        self,
        phonemes: list[str],
        threshold: float = 0.7
    ) -> list[tuple[str, float]]:
        """Find words phonetically similar to given phonemes.

        Args:
            phonemes: Query phoneme sequence
            threshold: Minimum similarity

        Returns:
            List of (word, similarity) tuples
        """
        similar = []
        query_key = " ".join(phonemes)

        for phon_key, words in self.reverse_index.items():
            candidate_phonemes = phon_key.split()
            sim = self.phoneme_similarity(phonemes, candidate_phonemes)

            if threshold <= sim < 1.0:  # Similar but not identical
                for word in words:
                    similar.append((word, sim))

        return sorted(similar, key=lambda x: -x[1])[:20]

    def detect(
        self,
        word: str,
        word_phonemes: list[str],
        full_sentence: str
    ) -> Optional[MalapropismMatch]:
        """Detect if word is a malapropism.

        Args:
            word: The spoken word
            word_phonemes: ARPAbet phoneme sequence
            full_sentence: Complete sentence for context

        Returns:
            MalapropismMatch if detected
        """
        if not self.perplexity_fn:
            # Without perplexity function, use simple detection
            return self._detect_simple(word, word_phonemes, full_sentence)

        # With perplexity function
        original_perplexity = self.perplexity_fn(full_sentence)

        alternatives = self.find_phonetically_similar(word_phonemes)

        best_replacement = None
        best_improvement = 0

        for alt_word, phon_sim in alternatives:
            modified_sentence = full_sentence.replace(word, alt_word, 1)
            new_perplexity = self.perplexity_fn(modified_sentence)

            improvement = original_perplexity - new_perplexity

            if improvement > best_improvement:
                best_improvement = improvement
                best_replacement = (alt_word, phon_sim, new_perplexity)

        if best_replacement and best_improvement > 10:
            return MalapropismMatch(
                spoken_word=word,
                intended_word=best_replacement[0],
                phonetic_similarity=best_replacement[1],
                coherence_improvement=best_improvement,
                confidence=min(best_improvement / 50.0, 0.9),
                type="malapropism"
            )

        return None

    def _detect_simple(
        self,
        word: str,
        word_phonemes: list[str],
        full_sentence: str
    ) -> Optional[MalapropismMatch]:
        """Simple malapropism detection without language model.

        Looks for common malapropism patterns.
        """
        # Common malapropisms dictionary
        COMMON_MALAPROPISMS = {
            "intensive": "intents and",
            "supposably": "supposedly",
            "expresso": "espresso",
            "excetera": "et cetera",
            "irregardless": "regardless",
            "pacifically": "specifically",
            "excape": "escape",
            "probly": "probably",
            "nucular": "nuclear",
            "expecially": "especially",
            "libary": "library",
            "febuary": "february",
            "pronounciation": "pronunciation",
            "mischievious": "mischievous",
        }

        word_lower = word.lower()
        if word_lower in COMMON_MALAPROPISMS:
            return MalapropismMatch(
                spoken_word=word,
                intended_word=COMMON_MALAPROPISMS[word_lower],
                phonetic_similarity=0.85,
                coherence_improvement=20.0,
                confidence=0.8,
                type="malapropism"
            )

        return None


# =============================================================================
# 17. MONDEGREEN DETECTOR
# =============================================================================

class MondegreenDetector:
    """Detect mondegreens - misheard lyrics/phrases.

    A mondegreen is when a phrase is misheard as something
    phonetically similar but semantically different.
    """

    def __init__(
        self,
        oronym_detector: OronymDetector,
        word_frequencies: Optional[dict[str, int]] = None
    ):
        """
        Args:
            oronym_detector: For finding alternative segmentations
            word_frequencies: Optional word frequency dict for plausibility
        """
        self.oronym_detector = oronym_detector
        self.word_frequencies = word_frequencies or {}

    def detect(
        self,
        phoneme_sequence: list[str],
        original_words: list[str]
    ) -> list[MondegreenMatch]:
        """Find plausible alternative hearings.

        Args:
            phoneme_sequence: Full phoneme sequence
            original_words: Original word sequence

        Returns:
            List of possible mondegreens
        """
        # Get all possible word segmentations
        alternatives = self.oronym_detector.detect(phoneme_sequence)

        original_text = " ".join(original_words).lower()

        results = []
        for alt in alternatives:
            alt_text = " ".join(alt.segmentation).lower()

            # Skip if same as original
            if alt_text == original_text:
                continue

            # Check if alternative is semantically plausible
            if self._is_plausible(alt.segmentation):
                results.append(MondegreenMatch(
                    original_text=original_words,
                    misheard_as=alt.segmentation,
                    phonetic_match=alt.score,
                    plausibility=self._calculate_plausibility(alt.segmentation),
                    type="mondegreen"
                ))

        return sorted(results, key=lambda x: -x.plausibility)[:5]

    def _is_plausible(self, words: list[str]) -> bool:
        """Check if word sequence is plausible English.

        Args:
            words: Proposed word sequence

        Returns:
            True if all words are common enough
        """
        for word in words:
            freq = self.word_frequencies.get(word.lower(), 0)
            if freq < 100:  # Very rare word
                return False
        return True

    def _calculate_plausibility(self, words: list[str]) -> float:
        """Score how plausible this hearing is.

        Args:
            words: Proposed word sequence

        Returns:
            Plausibility score (0-1)
        """
        if not words:
            return 0.0

        freq_score = sum(
            math.log(self.word_frequencies.get(w.lower(), 1) + 1)
            for w in words
        )
        return min(freq_score / len(words) / 10.0, 1.0)

    def detect_known_mondegreens(
        self,
        text: str
    ) -> Optional[MondegreenMatch]:
        """Check against database of known mondegreens.

        Args:
            text: Input text to check

        Returns:
            MondegreenMatch if known mondegreen found
        """
        # Famous mondegreens database
        KNOWN_MONDEGREENS = {
            "kiss the sky": {
                "misheard": ["kiss", "this", "guy"],
                "song": "Purple Haze - Jimi Hendrix"
            },
            "wrapped up like a douche": {
                "misheard": ["wrapped", "up", "like", "a", "douche"],
                "original": ["revved", "up", "like", "a", "deuce"],
                "song": "Blinded by the Light - Manfred Mann"
            },
            "hold me closer tony danza": {
                "misheard": ["hold", "me", "closer", "tony", "danza"],
                "original": ["hold", "me", "closer", "tiny", "dancer"],
                "song": "Tiny Dancer - Elton John"
            },
            "excuse me while i kiss this guy": {
                "original": ["excuse", "me", "while", "i", "kiss", "the", "sky"],
                "song": "Purple Haze - Jimi Hendrix"
            },
            "there's a bathroom on the right": {
                "original": ["there's", "a", "bad", "moon", "on", "the", "rise"],
                "song": "Bad Moon Rising - CCR"
            },
        }

        text_lower = text.lower()
        for misheard, info in KNOWN_MONDEGREENS.items():
            if misheard in text_lower:
                original = info.get("original", misheard.split())
                misheard_words = info.get("misheard", misheard.split())

                return MondegreenMatch(
                    original_text=original,
                    misheard_as=misheard_words,
                    phonetic_match=0.95,
                    plausibility=0.9,
                    type="known_mondegreen"
                )

        return None
