"""
Detection module for wordplay devices.

Provides 21 wordplay device detectors organized into three categories:
- 13 Phonetic-only detectors
- 4 Phonetic + Semantic detectors
- 4 Music-specific detectors
"""

from src.detection.models import (
    DeviceType,
    WordplayResult,
    DetectionResult,
    HomophoneMatch,
    OronymMatch,
    RhymeMatch,
    AssonanceMatch,
    ConsonanceMatch,
    AlliterationMatch,
    InternalRhymeMatch,
    MultisyllabicRhymeMatch,
    CompoundRhymeMatch,
    OnomatopoeiaMatch,
    SoundTextureMatch,
    StackedRhymeMatch,
    PunMatch,
    DoubleEntendreMatch,
    MalapropismMatch,
    MondegreenMatch,
    PolyrhythmicMatch,
    BreathRhymeMatch,
    MelismaMatch,
    SampleFlipMatch,
)

from src.detection.phonetic_detectors import (
    HomophoneDetector,
    OronymDetector,
    PerfectRhymeDetector,
    SlantRhymeDetector,
    AssonanceDetector,
    ConsonanceDetector,
    AlliterationDetector,
    InternalRhymeDetector,
    MultisyllabicRhymeDetector,
    CompoundRhymeDetector,
    OnomatopoeiaDetector,
    EuphonyCacophonyDetector,
    StackedRhymeDetector,
)

from src.detection.semantic_detectors import (
    PunDetector,
    DoubleEntendreDetector,
    MalapropismDetector,
    MondegreenDetector,
)

from src.detection.music_detectors import (
    PolyrhythmicRhymeDetector,
    BreathRhymeDetector,
    MelismaDetector,
    SampleFlipDetector,
)

from src.detection.unified_detector import (
    UnifiedWordplayDetector,
    DetectorConfig,
)

__all__ = [
    # Models
    "DeviceType",
    "WordplayResult",
    "DetectionResult",
    "HomophoneMatch",
    "OronymMatch",
    "RhymeMatch",
    "AssonanceMatch",
    "ConsonanceMatch",
    "AlliterationMatch",
    "InternalRhymeMatch",
    "MultisyllabicRhymeMatch",
    "CompoundRhymeMatch",
    "OnomatopoeiaMatch",
    "SoundTextureMatch",
    "StackedRhymeMatch",
    "PunMatch",
    "DoubleEntendreMatch",
    "MalapropismMatch",
    "MondegreenMatch",
    "PolyrhythmicMatch",
    "BreathRhymeMatch",
    "MelismaMatch",
    "SampleFlipMatch",
    # Phonetic Detectors
    "HomophoneDetector",
    "OronymDetector",
    "PerfectRhymeDetector",
    "SlantRhymeDetector",
    "AssonanceDetector",
    "ConsonanceDetector",
    "AlliterationDetector",
    "InternalRhymeDetector",
    "MultisyllabicRhymeDetector",
    "CompoundRhymeDetector",
    "OnomatopoeiaDetector",
    "EuphonyCacophonyDetector",
    "StackedRhymeDetector",
    # Semantic Detectors
    "PunDetector",
    "DoubleEntendreDetector",
    "MalapropismDetector",
    "MondegreenDetector",
    # Music Detectors
    "PolyrhythmicRhymeDetector",
    "BreathRhymeDetector",
    "MelismaDetector",
    "SampleFlipDetector",
    # Unified
    "UnifiedWordplayDetector",
    "DetectorConfig",
]
