#!/usr/bin/env python3
"""
Script: scripts/seed_phonemes.py
Purpose: Seed IPA phoneme inventory into database
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.db_service import db_service
from src.models.phoneme import Phoneme, IPA_INVENTORY
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def seed_phonemes() -> int:
    """Seed IPA phoneme inventory into database.
    
    Returns:
        Number of phonemes inserted.
    """
    await db_service.initialize()
    
    count = 0
    async with db_service.session() as session:
        for symbol, category, description, example_word, example_ipa in IPA_INVENTORY:
            phoneme = Phoneme(
                symbol=symbol,
                ipa_category=category,
                description=description,
                example_word=example_word,
                example_ipa=example_ipa,
            )
            session.add(phoneme)
            count += 1
        
        await session.flush()
    
    await db_service.close()
    
    logger.info(f"Seeded {count} phonemes into database")
    return count


def main():
    """Main entry point."""
    asyncio.run(seed_phonemes())


if __name__ == "__main__":
    main()
