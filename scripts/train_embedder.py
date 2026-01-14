#!/usr/bin/env python3
"""
Script: scripts/train_embedder.py
Purpose: Train FastText phonetic embeddings per DELIV-008
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_fasttext(
    corpus_path: Path,
    output_path: Path,
    vector_dim: int = 64,
    min_ngram: int = 3,
    max_ngram: int = 6,
    epochs: int = 10,
) -> None:
    """Train FastText model on phonetic corpus.
    
    Args:
        corpus_path: Path to IPA corpus file (one IPA string per line).
        output_path: Path to save trained model.
        vector_dim: Embedding dimension.
        min_ngram: Minimum n-gram size.
        max_ngram: Maximum n-gram size.
        epochs: Training epochs.
    """
    try:
        from gensim.models import FastText
    except ImportError:
        logger.error("gensim not installed. Run: pip install gensim")
        return
    
    logger.info(f"Loading corpus from {corpus_path}")
    
    # Load corpus
    sentences = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            # Each line is an IPA string
            ipa = line.strip().strip("/[]")
            if ipa:
                # Treat each character as a "word" for character-level embeddings
                sentences.append(list(ipa))
    
    logger.info(f"Loaded {len(sentences)} IPA strings")
    
    # Train FastText
    logger.info(f"Training FastText (dim={vector_dim}, ngrams={min_ngram}-{max_ngram})")
    
    model = FastText(
        sentences=sentences,
        vector_size=vector_dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1,  # Skip-gram
        min_n=min_ngram,
        max_n=max_ngram,
        epochs=epochs,
    )
    
    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    logger.info(f"Model saved to {output_path}")
    
    # Print some stats
    vocab_size = len(model.wv)
    logger.info(f"Vocabulary size: {vocab_size}")


def prepare_corpus(input_file: Path, output_file: Path) -> None:
    """Prepare corpus file from database export.
    
    Args:
        input_file: TSV file with word and IPA columns.
        output_file: Output file with one IPA per line.
    """
    import csv
    
    logger.info(f"Preparing corpus from {input_file}")
    
    with open(input_file, "r", encoding="utf-8") as fin:
        with open(output_file, "w", encoding="utf-8") as fout:
            reader = csv.reader(fin, delimiter="\t")
            
            for row in reader:
                if len(row) >= 2:
                    ipa = row[1].strip()
                    fout.write(f"{ipa}\n")
    
    logger.info(f"Corpus prepared: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train FastText embeddings")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument(
        "corpus",
        type=Path,
        help="Path to IPA corpus file",
    )
    train_parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.model_dir) / "fasttext" / "phonetic.bin",
        help="Output model path",
    )
    train_parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="Embedding dimension",
    )
    train_parser.add_argument(
        "--min-ngram",
        type=int,
        default=3,
        help="Minimum n-gram size",
    )
    train_parser.add_argument(
        "--max-ngram",
        type=int,
        default=6,
        help="Maximum n-gram size",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs",
    )
    
    # Prepare command
    prep_parser = subparsers.add_parser("prepare", help="Prepare corpus file")
    prep_parser.add_argument(
        "input",
        type=Path,
        help="Input TSV file",
    )
    prep_parser.add_argument(
        "output",
        type=Path,
        help="Output corpus file",
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_fasttext(
            corpus_path=args.corpus,
            output_path=args.output,
            vector_dim=args.dim,
            min_ngram=args.min_ngram,
            max_ngram=args.max_ngram,
            epochs=args.epochs,
        )
    elif args.command == "prepare":
        prepare_corpus(args.input, args.output)


if __name__ == "__main__":
    main()
