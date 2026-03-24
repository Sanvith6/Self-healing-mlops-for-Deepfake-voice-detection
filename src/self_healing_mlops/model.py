from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

FEATURE_ORDER = [
    "duration",
    "mean_abs_amplitude",
    "rms",
    "peak_amplitude",
    "zero_crossing_rate",
    "spectral_centroid",
]


class PrototypeDistanceClassifier:
    """Simple baseline classifier using class feature prototypes."""

    def __init__(self) -> None:
        self.class_prototypes: dict[int, list[float]] = {}

    @staticmethod
    def _vectorize(feature_map: dict[str, float]) -> list[float]:
        return [float(feature_map.get(name, 0.0)) for name in FEATURE_ORDER]

    @staticmethod
    def _distance(left: Sequence[float], right: Sequence[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5

    def fit(self, feature_maps: Iterable[dict[str, float]], labels: Iterable[int]) -> None:
        grouped: dict[int, list[list[float]]] = {}
        for feature_map, label in zip(feature_maps, labels):
            grouped.setdefault(int(label), []).append(self._vectorize(feature_map))

        if len(grouped) < 2:
            raise ValueError("Training requires at least two classes")

        self.class_prototypes = {}
        for label, vectors in grouped.items():
            if not vectors:
                continue
            dims = len(vectors[0])
            self.class_prototypes[label] = [sum(v[d] for v in vectors) / len(vectors) for d in range(dims)]

    def predict(self, feature_map: dict[str, float]) -> int:
        if not self.class_prototypes:
            raise ValueError("Model is not fitted")
        vector = self._vectorize(feature_map)
        best_label = None
        best_distance = None
        for label, prototype in self.class_prototypes.items():
            dist = self._distance(vector, prototype)
            if best_distance is None or dist < best_distance:
                best_label = label
                best_distance = dist
        return int(best_label)

    def evaluate(self, feature_maps: Iterable[dict[str, float]], labels: Iterable[int]) -> float:
        pairs = list(zip(feature_maps, labels))
        if not pairs:
            return 0.0
        correct = sum(1 for feature_map, label in pairs if self.predict(feature_map) == int(label))
        return correct / len(pairs)

    def save(self, path: str | Path) -> None:
        payload = {"class_prototypes": self.class_prototypes, "feature_order": FEATURE_ORDER}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeDistanceClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        model.class_prototypes = {int(k): [float(v) for v in vals] for k, vals in payload["class_prototypes"].items()}
        return model
