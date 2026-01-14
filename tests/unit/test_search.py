"""
File: tests/unit/test_search.py
Purpose: Unit tests for search components
"""

import pytest
import numpy as np
from src.core.search.scorer import Scorer, ScoredCandidate
from src.core.search.embedder import PhoneticEmbedder


class TestScorer:
    """Tests for scoring functions."""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return Scorer()
    
    def test_score_identical_ipa(self, scorer):
        """Test scoring with identical IPA."""
        result = scorer.score(
            words=["hello"],
            candidate_ipa="/hɛloʊ/",
            query_ipa="/hɛloʊ/",
            frequencies=[100],
        )
        assert result.phonetic_similarity == 1.0
    
    def test_score_weights_sum_to_one(self, scorer):
        """Test that scoring weights sum to 1.0."""
        total = scorer.PHONETIC_WEIGHT + scorer.FREQUENCY_WEIGHT + scorer.CONTEXT_WEIGHT
        assert abs(total - 1.0) < 0.001
    
    def test_classify_oronym_type_perfect(self, scorer):
        """Test perfect oronym classification."""
        result = scorer.classify_oronym_type(1.0)
        assert result == "perfect"
    
    def test_classify_oronym_type_near_perfect(self, scorer):
        """Test near-perfect oronym classification."""
        result = scorer.classify_oronym_type(0.97)
        assert result == "near_perfect"
    
    def test_classify_oronym_type_close(self, scorer):
        """Test close oronym classification."""
        result = scorer.classify_oronym_type(0.92)
        assert result == "close"
    
    def test_classify_confidence_high(self, scorer):
        """Test high confidence classification."""
        result = scorer.classify_confidence(0.90)
        assert result == "high"
    
    def test_classify_confidence_medium(self, scorer):
        """Test medium confidence classification."""
        result = scorer.classify_confidence(0.75)
        assert result == "medium"
    
    def test_classify_confidence_low(self, scorer):
        """Test low confidence classification."""
        result = scorer.classify_confidence(0.50)
        assert result == "low"
    
    def test_rank_candidates_sorts_by_score(self, scorer):
        """Test that candidates are ranked by score."""
        candidates = [
            ScoredCandidate(
                words=["low"],
                ipa="/loʊ/",
                phonetic_similarity=0.5,
                frequency_weight=0.5,
                context_coherence=0.5,
                final_score=0.5,
            ),
            ScoredCandidate(
                words=["high"],
                ipa="/haɪ/",
                phonetic_similarity=0.9,
                frequency_weight=0.9,
                context_coherence=0.9,
                final_score=0.9,
            ),
        ]
        ranked = scorer.rank_candidates(candidates)
        assert ranked[0].final_score > ranked[1].final_score
    
    def test_deduplicate_removes_duplicates(self, scorer):
        """Test that deduplication removes duplicates."""
        candidates = [
            ScoredCandidate(
                words=["hello"],
                ipa="/hɛloʊ/",
                phonetic_similarity=0.9,
                frequency_weight=0.5,
                context_coherence=0.5,
                final_score=0.9,
            ),
            ScoredCandidate(
                words=["Hello"],  # Same word, different case
                ipa="/hɛloʊ/",
                phonetic_similarity=0.8,
                frequency_weight=0.5,
                context_coherence=0.5,
                final_score=0.8,
            ),
        ]
        deduped = scorer.deduplicate(candidates)
        assert len(deduped) == 1
        # Keeps first (highest score)
        assert deduped[0].final_score == 0.9


class TestPhoneticEmbedder:
    """Tests for phonetic embedder."""
    
    def test_hash_embed_returns_correct_dimension(self):
        """Test that hash embedding returns correct dimension."""
        embedder = PhoneticEmbedder()
        embedder._initialized = True  # Skip actual initialization
        
        result = embedder._hash_embed("hello")
        assert result.shape == (64,)
    
    def test_hash_embed_deterministic(self):
        """Test that hash embedding is deterministic."""
        embedder = PhoneticEmbedder()
        embedder._initialized = True
        
        result1 = embedder._hash_embed("hello")
        result2 = embedder._hash_embed("hello")
        np.testing.assert_array_equal(result1, result2)
    
    def test_hash_embed_different_for_different_inputs(self):
        """Test that different inputs produce different embeddings."""
        embedder = PhoneticEmbedder()
        embedder._initialized = True
        
        result1 = embedder._hash_embed("hello")
        result2 = embedder._hash_embed("world")
        assert not np.array_equal(result1, result2)
