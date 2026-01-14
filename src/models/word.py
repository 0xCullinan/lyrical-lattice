"""
File: src/models/word.py
Purpose: SQLAlchemy ORM model for words table per Section 7.1
"""

from sqlalchemy import String, Integer, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base, TimestampMixin


class Word(Base, TimestampMixin):
    """Word entry in the phonetic corpus.
    
    Stores word-to-IPA mappings with metadata including language,
    frequency, source, and FAISS vector index.
    
    Attributes:
        id: Primary key.
        word: The word or phrase (UTF-8, 1-100 chars).
        ipa: IPA transcription (1-200 chars).
        language: ISO 639-1 code with optional region (e.g., 'en_US').
        frequency: Word frequency in corpus, non-negative.
        source: Data source ('wikipron', 'genius', 'manual', 'generated').
        vector_id: Index in FAISS vector store.
    """
    
    __tablename__ = "words"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    word: Mapped[str] = mapped_column(String(100), nullable=False)
    ipa: Mapped[str] = mapped_column(String(200), nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en_US")
    frequency: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_id: Mapped[int] = mapped_column(Integer, nullable=False)
    
    __table_args__ = (
        UniqueConstraint("word", "ipa", "language", name="uq_word_ipa_language"),
        Index("idx_words_word", "word"),
        Index("idx_words_ipa", "ipa"),
        Index("idx_words_language", "language"),
        Index("idx_words_frequency", "frequency", postgresql_ops={"frequency": "DESC"}),
        Index("idx_words_vector_id", "vector_id"),
    )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Word(id={self.id}, word='{self.word}', ipa='{self.ipa}', lang='{self.language}')"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses.
        
        Returns:
            Dictionary representation of the word.
        """
        return {
            "id": self.id,
            "word": self.word,
            "ipa": self.ipa,
            "language": self.language,
            "frequency": self.frequency,
            "source": self.source,
            "vector_id": self.vector_id,
        }
