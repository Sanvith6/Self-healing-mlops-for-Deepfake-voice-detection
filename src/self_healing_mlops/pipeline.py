from __future__ import annotations

from pathlib import Path

from .audio import extract_features_from_wav
from .model import PrototypeDistanceClassifier


def collect_labeled_features(data_dir: str | Path) -> tuple[list[dict[str, float]], list[int], list[Path]]:
    """Collect features from data_dir/{real,fake}/*.wav where real=0 and fake=1."""
    base = Path(data_dir)
    features: list[dict[str, float]] = []
    labels: list[int] = []
    files: list[Path] = []

    for label_name, label in (("real", 0), ("fake", 1)):
        folder = base / label_name
        if not folder.exists():
            continue
        for wav_file in sorted(folder.rglob("*.wav")):
            features.append(extract_features_from_wav(wav_file))
            labels.append(label)
            files.append(wav_file)

    return features, labels, files


def train_model(data_dir: str | Path, model_path: str | Path) -> dict[str, float | int]:
    features, labels, _ = collect_labeled_features(data_dir)
    if len(features) < 2:
        raise ValueError("Need at least two samples to train")

    model = PrototypeDistanceClassifier()
    model.fit(features, labels)
    accuracy = model.evaluate(features, labels)
    model.save(model_path)
    return {"samples": len(features), "train_accuracy": accuracy}


def predict_file(model_path: str | Path, wav_path: str | Path) -> dict[str, int | str]:
    model = PrototypeDistanceClassifier.load(model_path)
    prediction = model.predict(extract_features_from_wav(wav_path))
    label_name = "fake" if prediction == 1 else "real"
    return {"prediction": prediction, "label": label_name}


def evaluate_model(data_dir: str | Path, model_path: str | Path) -> dict[str, float | int]:
    features, labels, _ = collect_labeled_features(data_dir)
    model = PrototypeDistanceClassifier.load(model_path)
    accuracy = model.evaluate(features, labels)
    return {"samples": len(features), "accuracy": accuracy}
