"""
File: src/models/phoneme.py
Purpose: SQLAlchemy ORM model for phonemes table per Section 7.1
"""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base


class Phoneme(Base):
    """IPA phoneme inventory reference.
    
    Stores metadata about each phoneme for reference and validation.
    
    Attributes:
        id: Primary key.
        symbol: IPA symbol (e.g., 'p', 'b', 'æ').
        ipa_category: Phonetic category (e.g., 'plosive', 'vowel').
        description: Human-readable description.
        example_word: Example word containing the phoneme.
        example_ipa: IPA transcription of example word.
    """
    
    __tablename__ = "phonemes"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False, unique=True)
    ipa_category: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    example_word: Mapped[str | None] = mapped_column(Text, nullable=True)
    example_ipa: Mapped[str | None] = mapped_column(Text, nullable=True)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Phoneme(symbol='{self.symbol}', category='{self.ipa_category}')"
    
    def to_dict(self) -> dict:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation.
        """
        return {
            "symbol": self.symbol,
            "category": self.ipa_category,
            "description": self.description,
            "example_word": self.example_word,
            "example_ipa": self.example_ipa,
        }


# IPA phoneme inventory for pre-populating the database
IPA_INVENTORY = [
    # Plosives
    ("p", "plosive", "Voiceless bilabial plosive", "pat", "/pæt/"),
    ("b", "plosive", "Voiced bilabial plosive", "bat", "/bæt/"),
    ("t", "plosive", "Voiceless alveolar plosive", "tap", "/tæp/"),
    ("d", "plosive", "Voiced alveolar plosive", "dad", "/dæd/"),
    ("k", "plosive", "Voiceless velar plosive", "cat", "/kæt/"),
    ("ɡ", "plosive", "Voiced velar plosive", "go", "/ɡoʊ/"),
    ("ʔ", "plosive", "Glottal stop", "uh-oh", "/ʔʌʔoʊ/"),
    
    # Nasals
    ("m", "nasal", "Bilabial nasal", "mom", "/mɑm/"),
    ("n", "nasal", "Alveolar nasal", "noon", "/nuːn/"),
    ("ŋ", "nasal", "Velar nasal", "sing", "/sɪŋ/"),
    
    # Fricatives
    ("f", "fricative", "Voiceless labiodental fricative", "fun", "/fʌn/"),
    ("v", "fricative", "Voiced labiodental fricative", "van", "/væn/"),
    ("θ", "fricative", "Voiceless dental fricative", "think", "/θɪŋk/"),
    ("ð", "fricative", "Voiced dental fricative", "this", "/ðɪs/"),
    ("s", "fricative", "Voiceless alveolar fricative", "see", "/siː/"),
    ("z", "fricative", "Voiced alveolar fricative", "zoo", "/zuː/"),
    ("ʃ", "fricative", "Voiceless postalveolar fricative", "she", "/ʃiː/"),
    ("ʒ", "fricative", "Voiced postalveolar fricative", "measure", "/ˈmɛʒər/"),
    ("h", "fricative", "Voiceless glottal fricative", "hat", "/hæt/"),
    
    # Affricates
    ("tʃ", "affricate", "Voiceless postalveolar affricate", "church", "/tʃɜːrtʃ/"),
    ("dʒ", "affricate", "Voiced postalveolar affricate", "judge", "/dʒʌdʒ/"),
    
    # Approximants
    ("ɹ", "approximant", "Alveolar approximant", "run", "/ɹʌn/"),
    ("j", "approximant", "Palatal approximant", "yes", "/jɛs/"),
    ("w", "approximant", "Labial-velar approximant", "win", "/wɪn/"),
    ("l", "approximant", "Alveolar lateral approximant", "let", "/lɛt/"),
    
    # Vowels - Monophthongs
    ("iː", "vowel", "Close front unrounded vowel (long)", "see", "/siː/"),
    ("ɪ", "vowel", "Near-close near-front unrounded vowel", "sit", "/sɪt/"),
    ("e", "vowel", "Close-mid front unrounded vowel", "bed", "/bɛd/"),
    ("ɛ", "vowel", "Open-mid front unrounded vowel", "bed", "/bɛd/"),
    ("æ", "vowel", "Near-open front unrounded vowel", "cat", "/kæt/"),
    ("ɑː", "vowel", "Open back unrounded vowel (long)", "father", "/ˈfɑːðər/"),
    ("ɒ", "vowel", "Open back rounded vowel", "lot", "/lɒt/"),
    ("ɔː", "vowel", "Open-mid back rounded vowel (long)", "thought", "/θɔːt/"),
    ("ʊ", "vowel", "Near-close near-back rounded vowel", "put", "/pʊt/"),
    ("uː", "vowel", "Close back rounded vowel (long)", "blue", "/bluː/"),
    ("ʌ", "vowel", "Open-mid back unrounded vowel", "cup", "/kʌp/"),
    ("ə", "vowel", "Mid central vowel (schwa)", "about", "/əˈbaʊt/"),
    ("ɜː", "vowel", "Open-mid central unrounded vowel (long)", "bird", "/bɜːrd/"),
    
    # Vowels - Diphthongs
    ("eɪ", "diphthong", "Close-mid front to close front", "say", "/seɪ/"),
    ("aɪ", "diphthong", "Open front to close front", "my", "/maɪ/"),
    ("ɔɪ", "diphthong", "Open-mid back to close front", "boy", "/bɔɪ/"),
    ("aʊ", "diphthong", "Open front to close back", "how", "/haʊ/"),
    ("oʊ", "diphthong", "Close-mid back to close back", "go", "/ɡoʊ/"),
    ("ɪə", "diphthong", "Near-close front to mid central", "near", "/nɪər/"),
    ("eə", "diphthong", "Close-mid front to mid central", "square", "/skweər/"),
    ("ʊə", "diphthong", "Near-close back to mid central", "cure", "/kjʊər/"),
    
    # Suprasegmentals
    ("ˈ", "suprasegmental", "Primary stress", "about", "/əˈbaʊt/"),
    ("ˌ", "suprasegmental", "Secondary stress", "understand", "/ˌʌndərˈstænd/"),
    ("ː", "suprasegmental", "Length mark", "see", "/siː/"),
    ("ˑ", "suprasegmental", "Half-length mark", None, None),
]
