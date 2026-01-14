"""
File: src/core/search/embedder.py
Purpose: FastText character n-gram embeddings per REQ-SEARCH-002, REQ-SEARCH-012
"""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger(__name__)


class PhoneticEmbedder:
    """FastText-based phonetic embedder.
    
    Generates 64-dimensional embeddings for IPA strings using
    character n-grams (3-6 characters) per REQ-SEARCH-002.
    
    Attributes:
        model: FastText model.
        dim: Embedding dimension (64).
    """
    
    DIMENSION = 64  # Per REQ-SEARCH-002
    MIN_NGRAM = 3   # Per REQ-SEARCH-012
    MAX_NGRAM = 6   # Per REQ-SEARCH-012
    
    def __init__(self):
        """Initialize embedder."""
        self._model = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize FastText model.
        
        Loads pre-trained model from disk or creates a simple
        character embedding fallback for development.
        """
        logger.info("Initializing phonetic embedder")
        
        model_path = Path(settings.model_dir) / "fasttext" / "phonetic.bin"
        
        if model_path.exists():
            try:
                from gensim.models import FastText
                self._model = FastText.load(str(model_path))
                logger.info(f"Loaded FastText model from {model_path}")
                metrics.set_model_loaded("fasttext", True)
            except Exception as e:
                logger.warning(f"Failed to load FastText model: {e}")
                self._init_fallback()
        else:
            logger.info("FastText model not found, using character embedding fallback")
            self._init_fallback()
        
        self._initialized = True
    
    def _init_fallback(self) -> None:
        """Initialize character-level embedding fallback.
        
        Creates a simple hash-based embedding when FastText is unavailable.
        """
        self._model = None
        logger.info("Using hash-based embedding fallback")
    
    def embed(self, ipa_string: str) -> np.ndarray:
        """Generate embedding for IPA string.
        
        Args:
            ipa_string: IPA transcription to embed.
            
        Returns:
            Normalized 64-dimensional embedding.
        """
        if not self._initialized:
            raise RuntimeError("Embedder not initialized")
        
        # Clean input
        cleaned = ipa_string.strip().strip("/[]")
        
        if self._model is not None:
            # Use FastText model
            vector = self._model.wv.get_vector(cleaned).astype(np.float32)
        else:
            # Use fallback character n-gram hashing
            vector = self._hash_embed(cleaned)
        
        # L2 normalize per REQ-SEARCH-003
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.astype(np.float32)
    
    def _hash_embed(self, text: str) -> np.ndarray:
        """Generate embedding using character n-gram hashing.
        
        A simple fallback when FastText is not available.
        Uses feature hashing on character n-grams.
        
        Args:
            text: Text to embed.
            
        Returns:
            64-dimensional embedding.
        """
        vector = np.zeros(self.DIMENSION, dtype=np.float32)
        
        # Generate n-grams
        for n in range(self.MIN_NGRAM, self.MAX_NGRAM + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # Hash n-gram to bucket
                h = hash(ngram)
                bucket = h % self.DIMENSION
                sign = 1 if (h >> 31) & 1 else -1
                vector[bucket] += sign
        
        return vector
    
    def embed_batch(self, ipa_strings: list[str]) -> np.ndarray:
        """Generate embeddings for multiple IPA strings.
        
        Args:
            ipa_strings: List of IPA transcriptions.
            
        Returns:
            Array of shape (n, 64) with normalized embeddings.
        """
        embeddings = []
        for ipa in ipa_strings:
            embeddings.append(self.embed(ipa))
        return np.vstack(embeddings)
    
    def similarity(self, ipa1: str, ipa2: str) -> float:
        """Calculate cosine similarity between two IPA strings.
        
        Args:
            ipa1: First IPA string.
            ipa2: Second IPA string.
            
        Returns:
            Cosine similarity (0.0 to 1.0).
        """
        v1 = self.embed(ipa1)
        v2 = self.embed(ipa2)
        
        # Vectors are already normalized, so dot product = cosine similarity
        return float(np.dot(v1, v2))
    
    def is_loaded(self) -> bool:
        """Check if embedder is initialized.
        
        Returns:
            True if ready.
        """
        return self._initialized
    
    def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            64.
        """
        return self.DIMENSION
