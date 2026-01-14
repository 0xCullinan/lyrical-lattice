"""
File: src/core/search/faiss_engine.py
Purpose: FAISS IndexFlatIP wrapper per REQ-SEARCH-001, REQ-SEARCH-004
"""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np

from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import metrics

logger = get_logger(__name__)


class FAISSEngine:
    """FAISS vector search engine.
    
    Uses IndexFlatIP (Inner Product) for exact nearest neighbor search.
    Designed for 10M+ vectors per REQ-SEARCH-001 with <5ms search per REQ-SEARCH-004.
    
    Attributes:
        index: FAISS index.
        word_mapping: Vector ID to word mapping.
    """
    
    def __init__(self):
        """Initialize FAISS engine."""
        self._index = None
        self._word_mapping: dict = {}
        self._ipa_mapping: dict = {}
        self._metadata: dict = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Load FAISS index and mappings from disk."""
        logger.info("Initializing FAISS engine")
        
        index_path = Path(settings.index_dir) / "words_en_us.faiss"
        mapping_path = Path(settings.index_dir) / "words_en_us.pkl"
        
        if index_path.exists() and mapping_path.exists():
            try:
                import faiss
                
                # Load index
                self._index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
                
                # Load mappings
                with open(mapping_path, "rb") as f:
                    data = pickle.load(f)
                
                self._word_mapping = {
                    i: word for i, word in enumerate(data["words"])
                }
                self._ipa_mapping = {
                    i: ipa for i, ipa in enumerate(data["ipa"])
                }
                self._metadata = data.get("metadata", {})
                
                metrics.set_corpus_size(self._index.ntotal)
                metrics.set_model_loaded("faiss", True)
                
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                self._init_empty_index()
        else:
            logger.info("FAISS index not found, initializing empty index")
            self._init_empty_index()
        
        self._initialized = True
        logger.info("FAISS engine initialized")
    
    def _init_empty_index(self) -> None:
        """Initialize an empty FAISS index."""
        try:
            import faiss
            
            # Create empty IndexFlatIP with 64 dimensions
            self._index = faiss.IndexFlatIP(64)
            logger.info("Created empty FAISS IndexFlatIP (64 dim)")
        except ImportError:
            logger.warning("FAISS not installed. Search will not work.")
            self._index = None
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search for nearest neighbors.
        
        Args:
            query_vector: Query embedding (64-dim, L2 normalized).
            k: Number of results to return.
            
        Returns:
            List of (vector_id, score) tuples sorted by score descending.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        
        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search
        scores, indices = self._index.search(query_vector.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # -1 indicates no result
                results.append((int(idx), float(score)))
        
        return results
    
    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int = 10,
    ) -> list[list[tuple[int, float]]]:
        """Batch search for nearest neighbors.
        
        Args:
            query_vectors: Query embeddings (n, 64).
            k: Number of results per query.
            
        Returns:
            List of result lists.
        """
        if self._index is None or self._index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        
        scores, indices = self._index.search(query_vectors.astype(np.float32), k)
        
        all_results = []
        for i in range(len(query_vectors)):
            results = []
            for score, idx in zip(scores[i], indices[i]):
                if idx != -1:
                    results.append((int(idx), float(score)))
            all_results.append(results)
        
        return all_results
    
    def get_word(self, vector_id: int) -> Optional[str]:
        """Get word by vector ID.
        
        Args:
            vector_id: FAISS vector index.
            
        Returns:
            Word string or None.
        """
        return self._word_mapping.get(vector_id)
    
    def get_ipa(self, vector_id: int) -> Optional[str]:
        """Get IPA by vector ID.
        
        Args:
            vector_id: FAISS vector index.
            
        Returns:
            IPA string or None.
        """
        return self._ipa_mapping.get(vector_id)
    
    def get_word_and_ipa(self, vector_id: int) -> tuple[Optional[str], Optional[str]]:
        """Get word and IPA by vector ID.
        
        Args:
            vector_id: FAISS vector index.
            
        Returns:
            Tuple of (word, ipa).
        """
        return self.get_word(vector_id), self.get_ipa(vector_id)
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        words: list[str],
        ipa_list: list[str],
    ) -> None:
        """Add vectors to the index.
        
        Args:
            vectors: Embeddings to add (n, 64).
            words: Corresponding words.
            ipa_list: Corresponding IPA transcriptions.
        """
        if self._index is None:
            raise RuntimeError("FAISS index not initialized")
        
        start_id = self._index.ntotal
        self._index.add(vectors.astype(np.float32))
        
        for i, (word, ipa) in enumerate(zip(words, ipa_list)):
            vid = start_id + i
            self._word_mapping[vid] = word
            self._ipa_mapping[vid] = ipa
        
        metrics.set_corpus_size(self._index.ntotal)
        logger.debug(f"Added {len(vectors)} vectors, total: {self._index.ntotal}")
    
    def save(self) -> None:
        """Save index and mappings to disk."""
        if self._index is None:
            return
        
        import faiss
        
        index_path = Path(settings.index_dir) / "words_en_us.faiss"
        mapping_path = Path(settings.index_dir) / "words_en_us.pkl"
        
        # Ensure directory exists
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self._index, str(index_path))
        
        # Save mappings
        words = [self._word_mapping.get(i, "") for i in range(self._index.ntotal)]
        ipa = [self._ipa_mapping.get(i, "") for i in range(self._index.ntotal)]
        
        data = {
            "words": words,
            "ipa": ipa,
            "vector_ids": list(range(self._index.ntotal)),
            "metadata": {
                "vector_dim": 64,
                "total_vectors": self._index.ntotal,
            },
        }
        
        with open(mapping_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved FAISS index to {index_path}")
    
    def get_size(self) -> int:
        """Get number of vectors in index.
        
        Returns:
            Number of vectors.
        """
        return self._index.ntotal if self._index else 0
    
    def is_loaded(self) -> bool:
        """Check if engine is initialized.
        
        Returns:
            True if ready.
        """
        return self._initialized
    
    def check_health(self) -> tuple[bool, Optional[int]]:
        """Check FAISS health.
        
        Returns:
            Tuple of (is_healthy, latency_ms).
        """
        import time
        
        if not self._index:
            return False, None
        
        try:
            start = time.perf_counter()
            # Do a simple search
            dummy = np.random.randn(1, 64).astype(np.float32)
            self._index.search(dummy, 1)
            latency_ms = int((time.perf_counter() - start) * 1000)
            return True, latency_ms
        except Exception as e:
            logger.error(f"FAISS health check failed: {e}")
            return False, None
