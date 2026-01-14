"""
File: src/utils/validators.py
Purpose: Custom validators for IPA, audio files, and other inputs per REQ-SEC-001, REQ-SEC-002
"""

import re
from pathlib import Path
from typing import Optional


# Valid IPA character ranges per Unicode IPA block (U+0250 to U+02AF)
# Plus common extensions and modifiers
IPA_PATTERN = re.compile(
    r'^[/\[\]]?'  # Optional delimiters
    r'['
    r'\u0020-\u007E'     # Basic ASCII (including space)
    r'\u00C0-\u00FF'     # Latin Extended
    r'\u0250-\u02AF'     # IPA Extensions
    r'\u0300-\u036F'     # Combining Diacriticals
    r'\u02B0-\u02FF'     # Spacing Modifier Letters
    r'\u1D00-\u1D7F'     # Phonetic Extensions
    r'\u1D80-\u1DBF'     # Phonetic Extensions Supplement
    r'ˈˌːˑ'              # Stress and length marks
    r'ʼ'                 # Ejective marker
    r']*'
    r'[/\[\]]?$'  # Optional delimiters
)

# Slang to phonetic mappings for REQ-G2P-006
SLANG_MAPPINGS = {
    "finna": "fixing to",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "tryna": "trying to",
    "boutta": "about to",
    "outta": "out of",
    "gimme": "give me",
    "lemme": "let me",
    "whatcha": "what are you",
    "gotcha": "got you",
    "dunno": "do not know",
    "ain't": "is not",
    "y'all": "you all",
    "imma": "i am going to",
    "ima": "i am going to",
}

# Emoji to word mappings for REQ-G2P-009
EMOJI_MAPPINGS = {
    "🔥": "fire",
    "💯": "hundred",
    "❤️": "love",
    "💔": "heartbreak",
    "😭": "crying",
    "😂": "laughing",
    "🙏": "pray",
    "💪": "strong",
    "👀": "eyes",
    "🎵": "music",
    "🎶": "music",
    "⭐": "star",
    "🌟": "star",
    "💀": "dead",
    "🤔": "thinking",
    "😤": "angry",
    "🥺": "pleading",
    "✨": "sparkle",
    "💕": "love",
    "🖤": "black heart",
}

# Audio file magic bytes for format validation
AUDIO_MAGIC_BYTES = {
    "wav": [
        (b"RIFF", 0, 4),  # RIFF header
        (b"WAVE", 8, 12),  # WAVE format
    ],
    "mp3": [
        (b"\xff\xfb", 0, 2),  # MPEG Audio Layer 3
        (b"\xff\xfa", 0, 2),  # MPEG Audio Layer 3
        (b"\xff\xf3", 0, 2),  # MPEG Audio Layer 3
        (b"\xff\xf2", 0, 2),  # MPEG Audio Layer 3
        (b"ID3", 0, 3),       # ID3 tag
    ],
    "m4a": [
        (b"ftyp", 4, 8),  # ftyp box
    ],
}


class ValidationError(Exception):
    """Custom validation error with context."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            message: Error message.
            field: Field that failed validation.
            value: Value that failed validation.
        """
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


def validate_ipa(ipa_string: str) -> bool:
    """Validate that a string contains only valid IPA characters.
    
    Args:
        ipa_string: String to validate.
        
    Returns:
        True if valid IPA.
        
    Raises:
        ValidationError: If string contains invalid IPA characters.
    """
    if not ipa_string or not ipa_string.strip():
        raise ValidationError(
            "IPA string cannot be empty",
            field="ipa",
            value=ipa_string,
        )
    
    # Check character by character for better error messages
    for idx, char in enumerate(ipa_string):
        # Skip whitespace and common delimiters
        if char in " /[]ˈˌːˑ":
            continue
        
        # Check if character is in valid IPA ranges
        code = ord(char)
        valid_ranges = [
            (0x0020, 0x007E),  # Basic ASCII
            (0x00C0, 0x00FF),  # Latin Extended
            (0x0250, 0x02AF),  # IPA Extensions
            (0x0300, 0x036F),  # Combining Diacriticals
            (0x02B0, 0x02FF),  # Spacing Modifier Letters
            (0x1D00, 0x1D7F),  # Phonetic Extensions
            (0x1D80, 0x1DBF),  # Phonetic Extensions Supplement
        ]
        
        is_valid = any(start <= code <= end for start, end in valid_ranges)
        if not is_valid:
            raise ValidationError(
                f"Invalid IPA character '{char}' (U+{code:04X}) at position {idx}",
                field="ipa",
                value=ipa_string,
            )
    
    return True


def validate_audio_file(file_path: Path, expected_extension: Optional[str] = None) -> str:
    """Validate audio file format by checking magic bytes.
    
    Args:
        file_path: Path to audio file.
        expected_extension: Optional expected file extension.
        
    Returns:
        Detected audio format (wav, mp3, m4a).
        
    Raises:
        ValidationError: If file is not a valid audio file or magic bytes don't match extension.
    """
    if not file_path.exists():
        raise ValidationError(
            f"File not found: {file_path}",
            field="audio",
            value=str(file_path),
        )
    
    # Read enough bytes for magic byte detection
    with open(file_path, "rb") as f:
        header = f.read(32)
    
    detected_format = None
    
    # Check each format's magic bytes
    for format_name, magic_patterns in AUDIO_MAGIC_BYTES.items():
        for pattern, start, end in magic_patterns:
            if header[start:end] == pattern:
                detected_format = format_name
                break
        if detected_format:
            break
    
    if not detected_format:
        raise ValidationError(
            "Unrecognized audio format. Supported formats: WAV, MP3, M4A",
            field="audio",
            value=str(file_path),
        )
    
    # Check if extension matches magic bytes (REQ-SEC-002)
    if expected_extension:
        expected_ext = expected_extension.lower().lstrip(".")
        if expected_ext != detected_format:
            raise ValidationError(
                f"Magic bytes indicate {detected_format.upper()} but extension is .{expected_ext}. "
                "File extension must match actual format.",
                field="audio",
                value=str(file_path),
            )
    
    return detected_format


def preprocess_text(text: str) -> str:
    """Preprocess text for G2P conversion.
    
    Handles slang terms (REQ-G2P-006), repeated characters (REQ-G2P-007),
    numbers (REQ-G2P-008), and emojis (REQ-G2P-009).
    
    Args:
        text: Raw input text.
        
    Returns:
        Preprocessed text ready for G2P.
    """
    if not text or not text.strip():
        raise ValidationError(
            "Text cannot be empty or only whitespace",
            field="text",
            value=text,
        )
    
    result = text.lower()
    
    # Replace emojis with words (REQ-G2P-009)
    for emoji, word in EMOJI_MAPPINGS.items():
        result = result.replace(emoji, f" {word} ")
    
    # Replace slang with full forms (REQ-G2P-006)
    words = result.split()
    processed_words = []
    for word in words:
        # Strip punctuation for lookup
        clean_word = word.strip(".,!?;:'\"")
        if clean_word in SLANG_MAPPINGS:
            processed_words.append(SLANG_MAPPINGS[clean_word])
        else:
            processed_words.append(word)
    result = " ".join(processed_words)
    
    # Handle repeated characters (REQ-G2P-007)
    # e.g., "skrrrt" -> "skrt", "heeey" -> "hey"
    result = re.sub(r'(.)\1{2,}', r'\1\1', result)
    
    # Handle numbers (REQ-G2P-008)
    # Common patterns like "4eva" -> "forever", "2nite" -> "tonight"
    number_replacements = {
        r'\b4eva\b': 'forever',
        r'\b4ever\b': 'forever',
        r'\b2nite\b': 'tonight',
        r'\b2night\b': 'tonight',
        r'\b2day\b': 'today',
        r'\b2morrow\b': 'tomorrow',
        r'\b4\b': 'for',
        r'\b2\b': 'to',
        r'\bu\b': 'you',
        r'\br\b': 'are',
        r'\bn\b': 'and',
        r'\bb4\b': 'before',
    }
    for pattern, replacement in number_replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    result = " ".join(result.split())
    
    return result


def validate_text_length(text: str, max_length: int = 512) -> str:
    """Validate text length per REQ-G2P-001.
    
    Args:
        text: Text to validate.
        max_length: Maximum allowed length.
        
    Returns:
        Validated text.
        
    Raises:
        ValidationError: If text exceeds maximum length.
    """
    if len(text) > max_length:
        raise ValidationError(
            f"Text exceeds maximum length of {max_length} characters",
            field="text",
            value=text[:50] + "...",
        )
    return text


def validate_phoneme_sequence_length(phonemes: list[str], max_length: int = 100) -> list[str]:
    """Validate phoneme sequence length per REQ-SEARCH-008.
    
    Args:
        phonemes: List of phonemes.
        max_length: Maximum allowed length.
        
    Returns:
        Validated phoneme list.
        
    Raises:
        ValidationError: If sequence exceeds maximum length.
    """
    if len(phonemes) > max_length:
        raise ValidationError(
            f"Phoneme sequence exceeds maximum length of {max_length}",
            field="phonemes",
            value=str(phonemes[:10]) + "...",
        )
    return phonemes
