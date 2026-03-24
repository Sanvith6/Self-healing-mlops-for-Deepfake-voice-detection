"""Self-healing MLOps deepfake voice detection package."""

from .audio import extract_features, read_wav_mono
from .model import PrototypeDistanceClassifier

__all__ = ["read_wav_mono", "extract_features", "PrototypeDistanceClassifier"]
