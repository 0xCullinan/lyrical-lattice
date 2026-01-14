"""
File: src/core/search/scorer.py
Purpose: Scoring functions for oronym and rhyme ranking per REQ-SEARCH-007, RULE-001 to RULE-006
"""

import math
from typing import Optional
from dataclasses import dataclass

from src.core.g2p.ipa_utils import IPAUtils


@dataclass
class ScoredCandidate:
    """A scored candidate for oronym or rhyme matching.
    
    Attributes:
        words: List of words in the candidate.
        ipa: Combined IPA transcription.
        phonetic_similarity: Phonetic similarity score (0-1).
        frequency_weight: Frequency-based weight (0-1).
        context_coherence: Context coherence score (0-1).
        final_score: Weighted final score.
    """
    words: list[str]
    ipa: str
    phonetic_similarity: float
    frequency_weight: float
    context_coherence: float
    final_score: float


class Scorer:
    """Scorer for ranking oronym and rhyme candidates.
    
    Implements scoring formula per REQ-SEARCH-007:
    score = (phonetic_similarity * 0.7) + (frequency_weight * 0.2) + (context_coherence * 0.1)
    """
    
    # Weights per REQ-SEARCH-007
    PHONETIC_WEIGHT = 0.7
    FREQUENCY_WEIGHT = 0.2
    CONTEXT_WEIGHT = 0.1
    
    # Thresholds per RULE-005
    PERFECT_THRESHOLD = 1.0
    NEAR_PERFECT_THRESHOLD = 0.95
    CLOSE_THRESHOLD = 0.90
    
    # Confidence thresholds per RULE-006
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.70
    
    def __init__(self, max_corpus_frequency: int = 1000000):
        """Initialize scorer.
        
        Args:
            max_corpus_frequency: Maximum frequency in corpus for normalization.
        """
        self.max_corpus_frequency = max_corpus_frequency
    
    def score(
        self,
        words: list[str],
        candidate_ipa: str,
        query_ipa: str,
        frequencies: list[int],
        context_embeddings: Optional[list] = None,
    ) -> ScoredCandidate:
        """Score a candidate against a query.
        
        Args:
            words: Candidate words.
            candidate_ipa: Candidate IPA.
            query_ipa: Query IPA.
            frequencies: Word frequencies.
            context_embeddings: Optional word embeddings for context scoring.
            
        Returns:
            ScoredCandidate with all scores.
        """
        # Calculate phonetic similarity per RULE-001
        phonetic_sim = IPAUtils.phonetic_similarity(candidate_ipa, query_ipa)
        
        # Calculate frequency weight per RULE-002
        avg_freq = sum(frequencies) / len(frequencies) if frequencies else 0
        freq_weight = self._calculate_frequency_weight(avg_freq)
        
        # Calculate context coherence per RULE-003
        context_coherence = self._calculate_context_coherence(
            words, context_embeddings
        )
        
        # Calculate final score per RULE-004 and REQ-SEARCH-007
        final_score = (
            phonetic_sim * self.PHONETIC_WEIGHT +
            freq_weight * self.FREQUENCY_WEIGHT +
            context_coherence * self.CONTEXT_WEIGHT
        )
        
        return ScoredCandidate(
            words=words,
            ipa=candidate_ipa,
            phonetic_similarity=phonetic_sim,
            frequency_weight=freq_weight,
            context_coherence=context_coherence,
            final_score=final_score,
        )
    
    def _calculate_frequency_weight(self, frequency: float) -> float:
        """Calculate frequency weight using log scaling per RULE-002.
        
        Args:
            frequency: Word frequency.
            
        Returns:
            Normalized frequency weight (0 to 1).
        """
        if frequency <= 0:
            return 0.0
        
        # log(frequency + 1) / log(max_frequency + 1)
        return math.log(frequency + 1) / math.log(self.max_corpus_frequency + 1)
    
    def _calculate_context_coherence(
        self,
        words: list[str],
        embeddings: Optional[list] = None,
    ) -> float:
        """Calculate context coherence per RULE-003.
        
        Uses cosine similarity between adjacent word embeddings.
        Returns 0.5 as default when embeddings are not available.
        
        Args:
            words: List of words.
            embeddings: Optional word embeddings.
            
        Returns:
            Context coherence score (0 to 1).
        """
        if not embeddings or len(embeddings) < 2:
            # Default coherence when no embeddings available
            return 0.5
        
        import numpy as np
        
        # Calculate average cosine similarity between adjacent words
        similarities = []
        for i in range(len(embeddings) - 1):
            e1 = np.array(embeddings[i])
            e2 = np.array(embeddings[i + 1])
            
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(e1, e2) / (norm1 * norm2)
                similarities.append(max(0, sim))  # Ensure non-negative
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def classify_oronym_type(self, phonetic_similarity: float) -> str:
        """Classify oronym type based on phonetic similarity per RULE-005.
        
        Args:
            phonetic_similarity: Phonetic similarity score.
            
        Returns:
            'perfect', 'near_perfect', or 'close'.
        """
        if phonetic_similarity >= self.PERFECT_THRESHOLD:
            return "perfect"
        elif phonetic_similarity >= self.NEAR_PERFECT_THRESHOLD:
            return "near_perfect"
        else:
            return "close"
    
    def classify_confidence(self, score: float) -> str:
        """Classify confidence level based on score per RULE-006.
        
        Args:
            score: Final score.
            
        Returns:
            'high', 'medium', or 'low'.
        """
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def classify_rhyme_type(
        self,
        query_ipa: str,
        candidate_ipa: str,
        syllables_matched: int,
    ) -> str:
        """Classify rhyme type per REQ-RHYME-002.
        
        Args:
            query_ipa: Query IPA.
            candidate_ipa: Candidate IPA.
            syllables_matched: Number of matching syllables.
            
        Returns:
            'perfect', 'near', 'assonance', 'consonance', or 'multisyllabic'.
        """
        if syllables_matched >= 2:
            return "multisyllabic"
        
        # Get rhyme portions
        q_rhyme = IPAUtils.extract_rhyme_portion(query_ipa)
        c_rhyme = IPAUtils.extract_rhyme_portion(candidate_ipa)
        
        # Tokenize
        q_phonemes = IPAUtils.tokenize(q_rhyme)
        c_phonemes = IPAUtils.tokenize(c_rhyme)
        
        if q_phonemes == c_phonemes:
            return "perfect"
        
        # Check vowel match (assonance)
        q_vowels = [p for p in q_phonemes if IPAUtils.is_vowel(p)]
        c_vowels = [p for p in c_phonemes if IPAUtils.is_vowel(p)]
        
        if q_vowels == c_vowels and q_vowels:
            return "assonance"
        
        # Check consonant match (consonance)
        q_cons = [p for p in q_phonemes if IPAUtils.is_consonant(p)]
        c_cons = [p for p in c_phonemes if IPAUtils.is_consonant(p)]
        
        if q_cons == c_cons and q_cons:
            return "consonance"
        
        # Calculate similarity for near rhyme
        similarity = IPAUtils.phonetic_similarity(q_rhyme, c_rhyme)
        if similarity >= 0.80:
            return "near"
        
        # Default
        return "near"
    
    def rank_candidates(
        self,
        candidates: list[ScoredCandidate],
        min_similarity: float = 0.0,
    ) -> list[ScoredCandidate]:
        """Rank and filter candidates per RULE-011, RULE-012.
        
        Args:
            candidates: List of scored candidates.
            min_similarity: Minimum phonetic similarity threshold.
            
        Returns:
            Sorted and filtered candidates.
        """
        # Filter by minimum similarity
        filtered = [
            c for c in candidates
            if c.phonetic_similarity >= min_similarity
        ]
        
        # Sort by final score descending per RULE-012
        filtered.sort(key=lambda c: c.final_score, reverse=True)
        
        return filtered
    
    def deduplicate(
        self,
        candidates: list[ScoredCandidate],
    ) -> list[ScoredCandidate]:
        """Deduplicate candidates by phrase per RULE-011.
        
        Args:
            candidates: List of candidates.
            
        Returns:
            Deduplicated list (keeps highest scoring).
        """
        seen = set()
        result = []
        
        for c in candidates:
            # Create case-insensitive key
            key = " ".join(w.lower() for w in c.words)
            if key not in seen:
                seen.add(key)
                result.append(c)
        
        return result
