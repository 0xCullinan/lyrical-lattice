"""
File: src/core/search/beam_search.py
Purpose: Sliding window beam search per REQ-SEARCH-006, RULE-007 to RULE-010
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.core.config import settings
from src.core.search.embedder import PhoneticEmbedder
from src.core.search.faiss_engine import FAISSEngine
from src.core.search.scorer import Scorer, ScoredCandidate
from src.core.g2p.ipa_utils import IPAUtils
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BeamHypothesis:
    """A hypothesis in beam search.
    
    Attributes:
        words: Words decoded so far.
        ipa_parts: IPA parts corresponding to words.
        phonemes_consumed: Number of phonemes consumed.
        score: Cumulative score.
    """
    words: list[str] = field(default_factory=list)
    ipa_parts: list[str] = field(default_factory=list)
    phonemes_consumed: int = 0
    score: float = 0.0
    
    @property
    def ipa(self) -> str:
        """Get combined IPA."""
        return " ".join(self.ipa_parts)
    
    def extend(
        self,
        word: str,
        ipa: str,
        phonemes: int,
        score_delta: float,
    ) -> "BeamHypothesis":
        """Create extended hypothesis.
        
        Args:
            word: New word.
            ipa: New IPA.
            phonemes: Number of phonemes in new word.
            score_delta: Score contribution.
            
        Returns:
            New extended hypothesis.
        """
        return BeamHypothesis(
            words=self.words + [word],
            ipa_parts=self.ipa_parts + [ipa],
            phonemes_consumed=self.phonemes_consumed + phonemes,
            score=self.score + score_delta,
        )


class BeamSearch:
    """Sliding window beam search for oronym detection.
    
    Implements beam search per REQ-SEARCH-006 with:
    - Beam width: 10
    - Max word length: 15 phonemes
    - Min word length: 1 phoneme
    - Overlap: 0-3 phonemes
    """
    
    def __init__(
        self,
        embedder: PhoneticEmbedder,
        faiss_engine: FAISSEngine,
        scorer: Scorer,
    ):
        """Initialize beam search.
        
        Args:
            embedder: Phonetic embedder.
            faiss_engine: FAISS search engine.
            scorer: Scoring function.
        """
        self.embedder = embedder
        self.faiss = faiss_engine
        self.scorer = scorer
        
        # Parameters per REQ-SEARCH-006
        self.beam_width = settings.beam_width  # Default 10
        self.max_word_length = 15
        self.min_word_length = 1
        self.max_overlap = 3
    
    def search(
        self,
        phonemes: list[str],
        query_ipa: str,
        max_results: int = 50,
    ) -> list[ScoredCandidate]:
        """Search for word sequences matching phoneme sequence.
        
        Args:
            phonemes: Input phoneme sequence.
            query_ipa: Original query IPA for scoring.
            max_results: Maximum results to return.
            
        Returns:
            List of scored candidates sorted by score.
        """
        if not phonemes:
            return []
        
        n = len(phonemes)
        
        # Initialize beam with empty hypothesis
        beam = [BeamHypothesis()]
        completed = []
        
        # Iterate until all hypotheses complete
        max_iterations = n * 2  # Safety limit
        iteration = 0
        
        while beam and iteration < max_iterations:
            iteration += 1
            
            new_beam = []
            
            for hyp in beam:
                start = hyp.phonemes_consumed
                
                if start >= n:
                    # Hypothesis complete
                    completed.append(hyp)
                    continue
                
                # Try different window sizes per RULE-007
                for window_size in range(self.min_word_length, 
                                        min(self.max_word_length, n - start) + 1):
                    end = start + window_size
                    window_phonemes = phonemes[start:end]
                    window_ipa = "".join(window_phonemes)
                    
                    # Get embedding and search
                    embedding = self.embedder.embed(window_ipa)
                    results = self.faiss.search(embedding, k=5)
                    
                    for vec_id, search_score in results:
                        word = self.faiss.get_word(vec_id)
                        word_ipa = self.faiss.get_ipa(vec_id)
                        
                        if word and word_ipa:
                            # Calculate phonetic similarity
                            similarity = IPAUtils.phonetic_similarity(
                                window_ipa, word_ipa
                            )
                            
                            # Only keep good matches
                            if similarity >= 0.7:
                                new_hyp = hyp.extend(
                                    word=word,
                                    ipa=word_ipa,
                                    phonemes=window_size,
                                    score_delta=similarity * search_score,
                                )
                                new_beam.append(new_hyp)
                    
                    # Try overlapping matches per RULE-008
                    for overlap in range(1, min(self.max_overlap + 1, window_size)):
                        overlap_start = end - overlap
                        if overlap_start <= start:
                            continue
                        
                        # This creates alternative split points
                        overlap_phonemes = phonemes[start:overlap_start]
                        if overlap_phonemes:
                            overlap_ipa = "".join(overlap_phonemes)
                            embedding = self.embedder.embed(overlap_ipa)
                            results = self.faiss.search(embedding, k=3)
                            
                            for vec_id, search_score in results:
                                word = self.faiss.get_word(vec_id)
                                word_ipa = self.faiss.get_ipa(vec_id)
                                
                                if word and word_ipa:
                                    similarity = IPAUtils.phonetic_similarity(
                                        overlap_ipa, word_ipa
                                    )
                                    
                                    if similarity >= 0.7:
                                        new_hyp = hyp.extend(
                                            word=word,
                                            ipa=word_ipa,
                                            phonemes=len(overlap_phonemes),
                                            score_delta=similarity * search_score * 0.9,
                                        )
                                        new_beam.append(new_hyp)
            
            # Prune beam per RULE-009
            new_beam.sort(key=lambda h: h.score, reverse=True)
            beam = new_beam[:self.beam_width]
        
        # Add remaining hypotheses to completed
        completed.extend(beam)
        
        # Convert to ScoredCandidates
        candidates = []
        for hyp in completed:
            if hyp.words:
                candidate = self.scorer.score(
                    words=hyp.words,
                    candidate_ipa=hyp.ipa,
                    query_ipa=query_ipa,
                    frequencies=[100] * len(hyp.words),  # Default frequency
                )
                candidates.append(candidate)
        
        # Deduplicate and rank
        candidates = self.scorer.deduplicate(candidates)
        candidates = self.scorer.rank_candidates(candidates, min_similarity=0.5)
        
        return candidates[:max_results]
    
    def search_single_word(
        self,
        ipa: str,
        k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Simple single-word search without beam.
        
        Args:
            ipa: Query IPA.
            k: Number of results.
            
        Returns:
            List of (word, ipa, score) tuples.
        """
        embedding = self.embedder.embed(ipa)
        results = self.faiss.search(embedding, k=k)
        
        output = []
        for vec_id, score in results:
            word = self.faiss.get_word(vec_id)
            word_ipa = self.faiss.get_ipa(vec_id)
            if word and word_ipa:
                output.append((word, word_ipa, float(score)))
        
        return output
    
    def find_oronyms(
        self,
        query_ipa: str,
        max_results: int = 50,
        min_similarity: float = 0.90,
    ) -> list[ScoredCandidate]:
        """Find oronyms for a query.
        
        Oronyms are phrases that sound identical or nearly identical
        per REQ-ORONYM-001.
        
        Args:
            query_ipa: Query IPA transcription.
            max_results: Maximum results.
            min_similarity: Minimum phonetic similarity per REQ-ORONYM-001.
            
        Returns:
            Oronym candidates with similarity >= min_similarity.
        """
        # Tokenize query
        phonemes = IPAUtils.tokenize(query_ipa)
        
        if not phonemes:
            return []
        
        # Validate per REQ-ORONYM-006: max 1 phoneme difference per 5 phonemes
        max_diff = max(1, len(phonemes) // 5)
        
        # Run beam search
        candidates = self.search(phonemes, query_ipa, max_results * 2)
        
        # Filter by similarity threshold
        oronyms = []
        for c in candidates:
            if c.phonetic_similarity >= min_similarity:
                # Check phoneme difference constraint
                c_phonemes = IPAUtils.tokenize(c.ipa)
                diff = abs(len(c_phonemes) - len(phonemes))
                
                if diff <= max_diff:
                    oronyms.append(c)
        
        return oronyms[:max_results]
    
    def find_rhymes(
        self,
        query_ipa: str,
        max_results: int = 100,
        min_similarity: float = 0.80,
    ) -> list[ScoredCandidate]:
        """Find rhyming words per REQ-RHYME-001.
        
        Args:
            query_ipa: Query word IPA.
            max_results: Maximum results.
            min_similarity: Minimum rhyme similarity.
            
        Returns:
            Rhyme candidates.
        """
        # Extract rhyme portion (from last stressed syllable)
        rhyme_portion = IPAUtils.extract_rhyme_portion(query_ipa)
        
        if not rhyme_portion:
            return []
        
        # Search for matches
        embedding = self.embedder.embed(rhyme_portion)
        results = self.faiss.search(embedding, k=max_results * 2)
        
        candidates = []
        for vec_id, score in results:
            word = self.faiss.get_word(vec_id)
            word_ipa = self.faiss.get_ipa(vec_id)
            
            if word and word_ipa:
                word_rhyme = IPAUtils.extract_rhyme_portion(word_ipa)
                similarity = IPAUtils.phonetic_similarity(rhyme_portion, word_rhyme)
                
                if similarity >= min_similarity:
                    candidate = self.scorer.score(
                        words=[word],
                        candidate_ipa=word_ipa,
                        query_ipa=query_ipa,
                        frequencies=[100],
                    )
                    candidate.phonetic_similarity = similarity
                    candidates.append(candidate)
        
        # Deduplicate and sort
        candidates = self.scorer.deduplicate(candidates)
        candidates.sort(key=lambda c: c.phonetic_similarity, reverse=True)
        
        return candidates[:max_results]
