"""
File: src/core/g2p/byt5_engine.py
Purpose: ByT5 model wrapper for text-to-IPA conversion per MODEL-001
"""

import time
from typing import Optional
from dataclasses import dataclass

from src.core.config import settings
from src.utils.logger import get_logger
from src.utils.metrics import metrics
from src.utils.validators import preprocess_text, ValidationError

logger = get_logger(__name__)


@dataclass
class G2PResult:
    """Result from G2P conversion.
    
    Attributes:
        ipa: Primary IPA transcription.
        confidence: Confidence score (0.0 to 1.0).
        alternatives: Alternative transcriptions with confidence.
        processing_time_ms: Processing time in milliseconds.
    """
    ipa: str
    confidence: float
    alternatives: list[tuple[str, float]]
    processing_time_ms: int


class ByT5Engine:
    """ByT5-based grapheme-to-phoneme engine.
    
    Uses google/byt5-small model for text-to-IPA conversion.
    Falls back to g2p-en library when ByT5 is not available.
    
    Satisfies requirements REQ-G2P-001 through REQ-G2P-012.
    """
    
    def __init__(self):
        """Initialize G2P engine."""
        self._model = None
        self._tokenizer = None
        self._g2p_fallback = None
        self._use_transformers = False
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the G2P model.
        
        Attempts to load ByT5 model, falls back to g2p-en if unavailable.
        """
        logger.info("Initializing G2P engine")
        
        # Try to load ByT5 model
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch
            
            model_name = "charsiu/g2p_multilingual_byT5_small_100"
            model_path = f"{settings.model_dir}/byt5_charsiu"
            
            # Check if model exists locally, otherwise use HuggingFace
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                logger.info(f"Loaded ByT5 from local path: {model_path}")
            except Exception:
                logger.info(f"Loading ByT5 from HuggingFace Hub: {model_name}")
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("ByT5 model loaded on GPU")
            # MPS disabled due to "Placeholder storage has not been allocated" error
            # elif torch.backends.mps.is_available():
            #     self._model = self._model.to("mps")
            #     logger.info("ByT5 model loaded on MPS (Apple Silicon)")
            else:
                logger.info("ByT5 model loaded on CPU")
            
            self._use_transformers = True
            metrics.set_model_loaded("byt5", True)
            
        except Exception as e:
            logger.warning(f"Failed to load ByT5 model: {e}. Using g2p-en fallback.")
            self._load_fallback()
        
        self._initialized = True
        logger.info("G2P engine initialized")
    
    def _load_fallback(self) -> None:
        """Load g2p-en as fallback."""
        try:
            from g2p_en import G2p
            self._g2p_fallback = G2p()
            metrics.set_model_loaded("g2p_en", True)
            logger.info("g2p-en fallback loaded")
        except ImportError:
            logger.error("g2p-en not installed. G2P will not work.")
            raise RuntimeError("No G2P backend available")
    
    async def phonemize(
        self,
        text: str,
        language: str = "en_US",
        max_alternatives: int = 5,
    ) -> G2PResult:
        """Convert text to IPA transcription.
        
        Args:
            text: Input text (1-512 characters per REQ-G2P-001).
            language: Source language code.
            max_alternatives: Maximum alternatives to return.
            
        Returns:
            G2PResult with IPA and confidence.
            
        Raises:
            ValidationError: If input is invalid.
            RuntimeError: If G2P engine not initialized.
        """
        if not self._initialized:
            raise RuntimeError("G2P engine not initialized")
        
        start_time = time.perf_counter()
        
        # Preprocess text (handles slang, emojis, repeated chars, numbers)
        processed_text = preprocess_text(text)
        
        # Perform G2P
        if self._use_transformers:
            ipa, confidence, alternatives = await self._byt5_phonemize(
                processed_text, max_alternatives
            )
        else:
            ipa, confidence, alternatives = self._fallback_phonemize(
                processed_text, max_alternatives
            )
        
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        metrics.record_g2p_latency(elapsed_ms / 1000)
        
        return G2PResult(
            ipa=ipa,
            confidence=confidence,
            alternatives=alternatives,
            processing_time_ms=elapsed_ms,
        )
    
    async def _byt5_phonemize(
        self,
        text: str,
        max_alternatives: int,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Phonemize using ByT5 model.
        
        Args:
            text: Preprocessed text.
            max_alternatives: Max alternatives.
            
        Returns:
            Tuple of (ipa, confidence, alternatives).
        """
        import torch
        
        # Prepare input - Charsiu/ByT5 usually expects just the text or <lang>: text
        # For simplicity trying direct text first
        prompt = f"<eng_us>: {text}"
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with beam search for alternatives
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=256,
                num_beams=max_alternatives + 1,
                num_return_sequences=min(max_alternatives + 1, 5),
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode outputs
        sequences = outputs.sequences
        results = []
        
        for seq in sequences:
            decoded = self._tokenizer.decode(seq, skip_special_tokens=True)
            # Clean up output
            decoded = decoded.strip()
            if not decoded.startswith("/"):
                decoded = f"/{decoded}/"
            results.append(decoded)
        
        # Calculate confidence from scores
        if hasattr(outputs, "sequences_scores"):
            scores = outputs.sequences_scores.softmax(dim=0).tolist()
        else:
            # Equal confidence if scores not available
            scores = [1.0 / len(results)] * len(results)
        
        primary_ipa = results[0] if results else "//"
        primary_confidence = scores[0] if scores else 0.5
        
        alternatives = []
        for i in range(1, len(results)):
            if scores[i] > 0.01:  # Only include if confidence > 1%
                alternatives.append((results[i], scores[i]))
        
        return primary_ipa, primary_confidence, alternatives
    
    def _fallback_phonemize(
        self,
        text: str,
        max_alternatives: int,
    ) -> tuple[str, float, list[tuple[str, float]]]:
        """Phonemize using g2p-en fallback.
        
        Args:
            text: Preprocessed text.
            max_alternatives: Max alternatives (not used in fallback).
            
        Returns:
            Tuple of (ipa, confidence, alternatives).
        """
        # g2p-en returns ARPABET, need to convert to IPA
        phonemes = self._g2p_fallback(text)
        
        # Convert ARPABET to IPA
        ipa_phonemes = [self._arpabet_to_ipa(p) for p in phonemes]
        ipa = "/" + "".join(ipa_phonemes) + "/"
        
        # g2p-en doesn't provide confidence/alternatives
        return ipa, 0.85, []
    
    def _arpabet_to_ipa(self, arpabet: str) -> str:
        """Convert ARPABET phoneme to IPA.
        
        Args:
            arpabet: ARPABET phoneme.
            
        Returns:
            IPA equivalent.
        """
        # ARPABET to IPA mapping
        mapping = {
            # Vowels
            "AA": "ɑ", "AA0": "ɑ", "AA1": "ˈɑ", "AA2": "ˌɑ",
            "AE": "æ", "AE0": "æ", "AE1": "ˈæ", "AE2": "ˌæ",
            "AH": "ʌ", "AH0": "ə", "AH1": "ˈʌ", "AH2": "ˌʌ",
            "AO": "ɔ", "AO0": "ɔ", "AO1": "ˈɔ", "AO2": "ˌɔ",
            "AW": "aʊ", "AW0": "aʊ", "AW1": "ˈaʊ", "AW2": "ˌaʊ",
            "AY": "aɪ", "AY0": "aɪ", "AY1": "ˈaɪ", "AY2": "ˌaɪ",
            "EH": "ɛ", "EH0": "ɛ", "EH1": "ˈɛ", "EH2": "ˌɛ",
            "ER": "ɜr", "ER0": "ər", "ER1": "ˈɜr", "ER2": "ˌɜr",
            "EY": "eɪ", "EY0": "eɪ", "EY1": "ˈeɪ", "EY2": "ˌeɪ",
            "IH": "ɪ", "IH0": "ɪ", "IH1": "ˈɪ", "IH2": "ˌɪ",
            "IY": "i", "IY0": "i", "IY1": "ˈi", "IY2": "ˌi",
            "OW": "oʊ", "OW0": "oʊ", "OW1": "ˈoʊ", "OW2": "ˌoʊ",
            "OY": "ɔɪ", "OY0": "ɔɪ", "OY1": "ˈɔɪ", "OY2": "ˌɔɪ",
            "UH": "ʊ", "UH0": "ʊ", "UH1": "ˈʊ", "UH2": "ˌʊ",
            "UW": "u", "UW0": "u", "UW1": "ˈu", "UW2": "ˌu",
            # Consonants
            "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
            "F": "f", "G": "g", "HH": "h", "JH": "dʒ",
            "K": "k", "L": "l", "M": "m", "N": "n",
            "NG": "ŋ", "P": "p", "R": "r", "S": "s",
            "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
            "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
        }
        
        # Handle space
        if arpabet == " ":
            return " "
        
        return mapping.get(arpabet.upper(), arpabet.lower())
    
    async def phonemize_batch(
        self,
        texts: list[str],
        language: str = "en_US",
    ) -> list[G2PResult]:
        """Batch phonemize multiple texts.
        
        Args:
            texts: List of input texts.
            language: Source language.
            
        Returns:
            List of G2PResult.
        """
        results = []
        for text in texts:
            result = await self.phonemize(text, language)
            results.append(result)
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded.
        
        Returns:
            True if initialized.
        """
        return self._initialized
    
    def get_backend(self) -> str:
        """Get active backend name.
        
        Returns:
            'byt5' or 'g2p_en'.
        """
        if self._use_transformers:
            return "byt5"
        return "g2p_en"
