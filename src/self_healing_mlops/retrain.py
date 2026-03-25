from __future__ import annotations

from collections.abc import Callable

from .monitor import SelfHealingMonitor


def maybe_retrain(
    monitor: SelfHealingMonitor,
    recent_accuracy: float,
    reference_feature_mean: float,
    current_feature_mean: float,
    retrain_callback: Callable[[], None],
) -> bool:
    """Update monitor state and invoke retraining callback when needed."""
    monitor.update(recent_accuracy, reference_feature_mean, current_feature_mean)
    if monitor.should_retrain():
        retrain_callback()
        return True
    return False
