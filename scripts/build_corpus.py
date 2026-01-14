#!/usr/bin/env python3
"""
Script: scripts/build_corpus.py
Purpose: Build phonetic corpus from WikiPron and other sources per DELIV-006
"""

import asyncio
import argparse
import csv
import sys
from pathlib import Path
from typing import AsyncGenerator

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.services.db_service import db_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def parse_wikipron(file_path: Path) -> AsyncGenerator[dict, None]:
    """Parse WikiPron TSV file.
    
    WikiPron format: word<TAB>ipa
    
    Args:
        file_path: Path to WikiPron TSV file.
        
    Yields:
        Word data dictionaries.
    """
    logger.info(f"Parsing WikiPron file: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        
        for line_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue
            
            word = row[0].strip()
            ipa = row[1].strip()
            
            if not word or not ipa:
                continue
            
            # Add slashes if not present
            if not ipa.startswith("/"):
                ipa = f"/{ipa}/"
            
            yield {
                "word": word,
                "ipa": ipa,
                "language": "en_US",
                "frequency": 0,
                "source": "wikipron",
                "vector_id": line_num - 1,
            }
            
            if line_num % 10000 == 0:
                logger.info(f"Processed {line_num} entries...")


async def parse_cmudict(file_path: Path) -> AsyncGenerator[dict, None]:
    """Parse CMU Pronouncing Dictionary file.
    
    CMUdict format: WORD  P R O N U N C I A T I O N
    
    Args:
        file_path: Path to CMUdict file.
        
    Yields:
        Word data dictionaries.
    """
    logger.info(f"Parsing CMUdict file: {file_path}")
    
    # ARPABET to IPA mapping
    arpabet_to_ipa = {
        "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
        "AY": "aɪ", "EH": "ɛ", "ER": "ɜr", "EY": "eɪ", "IH": "ɪ",
        "IY": "i", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
        "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
        "G": "g", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
        "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "r",
        "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
        "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
    }
    
    line_num = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith(";;;"):
                continue
            
            parts = line.split("  ")
            if len(parts) != 2:
                continue
            
            word = parts[0].strip()
            arpabet = parts[1].strip().split()
            
            # Convert ARPABET to IPA
            ipa_phonemes = []
            for phoneme in arpabet:
                # Remove stress markers
                base_phoneme = phoneme.rstrip("012")
                ipa = arpabet_to_ipa.get(base_phoneme, phoneme.lower())
                
                # Add stress marker
                if phoneme.endswith("1"):
                    ipa = "ˈ" + ipa
                elif phoneme.endswith("2"):
                    ipa = "ˌ" + ipa
                
                ipa_phonemes.append(ipa)
            
            ipa_str = f"/{''.join(ipa_phonemes)}/"
            
            yield {
                "word": word.lower(),
                "ipa": ipa_str,
                "language": "en_US",
                "frequency": 0,
                "source": "cmudict",
                "vector_id": line_num,
            }
            
            line_num += 1
            
            if line_num % 10000 == 0:
                logger.info(f"Processed {line_num} entries...")


async def build_corpus(
    input_files: list[Path],
    batch_size: int = 1000,
) -> int:
    """Build corpus from input files.
    
    Args:
        input_files: List of input file paths.
        batch_size: Batch size for database inserts.
        
    Returns:
        Total entries inserted.
    """
    await db_service.initialize()
    
    total_inserted = 0
    batch = []
    
    for file_path in input_files:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        # Determine parser based on file name
        if "wikipron" in file_path.name.lower():
            parser = parse_wikipron(file_path)
        elif "cmu" in file_path.name.lower():
            parser = parse_cmudict(file_path)
        else:
            logger.warning(f"Unknown file format: {file_path}")
            continue
        
        async for entry in parser:
            batch.append(entry)
            
            if len(batch) >= batch_size:
                count = await db_service.bulk_insert_words(batch)
                total_inserted += count
                batch = []
                logger.info(f"Inserted batch, total: {total_inserted}")
    
    # Insert remaining batch
    if batch:
        count = await db_service.bulk_insert_words(batch)
        total_inserted += count
    
    await db_service.close()
    
    logger.info(f"Corpus build complete. Total entries: {total_inserted}")
    return total_inserted


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build phonetic corpus")
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input files (WikiPron TSV or CMUdict)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for database inserts",
    )
    
    args = parser.parse_args()
    
    asyncio.run(build_corpus(args.files, args.batch_size))


if __name__ == "__main__":
    main()
