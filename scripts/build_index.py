#!/usr/bin/env python3
"""
Script: scripts/build_index.py
Purpose: Build FAISS index from database corpus per DELIV-007
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.search.embedder import PhoneticEmbedder
from src.core.search.faiss_engine import FAISSEngine
from src.services.db_service import db_service
from src.utils.logger import get_logger
import numpy as np

logger = get_logger(__name__)


async def build_index(
    language: str = "en_US",
    batch_size: int = 10000,
) -> int:
    """Build FAISS index from database corpus.
    
    Args:
        language: Language code to build index for.
        batch_size: Batch size for processing.
        
    Returns:
        Total vectors indexed.
    """
    # Initialize services
    await db_service.initialize()
    
    embedder = PhoneticEmbedder()
    await embedder.initialize()
    
    faiss_engine = FAISSEngine()
    await faiss_engine.initialize()
    
    # Get corpus size
    corpus_size = await db_service.get_corpus_size()
    logger.info(f"Building index for {corpus_size} entries")
    
    if corpus_size == 0:
        logger.warning("No entries in corpus. Run build_corpus.py first.")
        return 0
    
    # Process in batches
    total_indexed = 0
    offset = 0
    
    async with db_service.session() as session:
        from sqlalchemy import select
        from src.models.word import Word
        
        while offset < corpus_size:
            # Fetch batch
            result = await session.execute(
                select(Word)
                .where(Word.language == language)
                .offset(offset)
                .limit(batch_size)
            )
            words = list(result.scalars().all())
            
            if not words:
                break
            
            # Generate embeddings
            ipa_strings = [w.ipa for w in words]
            word_texts = [w.word for w in words]
            
            embeddings = embedder.embed_batch(ipa_strings)
            
            # Add to index
            faiss_engine.add_vectors(
                vectors=embeddings,
                words=word_texts,
                ipa_list=ipa_strings,
            )
            
            total_indexed += len(words)
            offset += batch_size
            
            logger.info(f"Indexed {total_indexed}/{corpus_size} entries")
    
    # Save index
    faiss_engine.save()
    
    await db_service.close()
    
    logger.info(f"Index build complete. Total vectors: {total_indexed}")
    return total_indexed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build FAISS index")
    parser.add_argument(
        "--language",
        type=str,
        default="en_US",
        help="Language code to build index for",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing",
    )
    
    args = parser.parse_args()
    
    asyncio.run(build_index(args.language, args.batch_size))


if __name__ == "__main__":
    main()
