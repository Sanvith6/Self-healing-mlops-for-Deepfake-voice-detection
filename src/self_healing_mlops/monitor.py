from __future__ import annotations

from collections import deque


class SelfHealingMonitor:
    """Tracks performance and drift to decide when retraining is needed."""

    def __init__(
        self,
        accuracy_threshold: float = 0.8,
        drift_threshold: float = 0.25,
        window_size: int = 10,
    ) -> None:
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self._recent_accuracy: deque[float] = deque(maxlen=window_size)
        self.latest_drift = 0.0

    @staticmethod
    def _normalized_shift(reference: float, current: float) -> float:
        denominator = abs(reference) + 1e-9
        return abs(current - reference) / denominator

    def update(self, recent_accuracy: float, reference_feature_mean: float, current_feature_mean: float) -> None:
        self._recent_accuracy.append(float(recent_accuracy))
        self.latest_drift = self._normalized_shift(reference_feature_mean, current_feature_mean)

    @property
    def mean_accuracy(self) -> float:
        if not self._recent_accuracy:
            return 1.0
        return sum(self._recent_accuracy) / len(self._recent_accuracy)

    def should_retrain(self) -> bool:
        return self.mean_accuracy < self.accuracy_threshold or self.latest_drift > self.drift_threshold

    def status(self) -> dict[str, float | bool]:
        return {
            "mean_accuracy": self.mean_accuracy,
            "latest_drift": self.latest_drift,
            "needs_retraining": self.should_retrain(),
        }
