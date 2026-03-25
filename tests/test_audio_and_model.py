from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

from self_healing_mlops.audio import extract_features_from_wav
from self_healing_mlops.monitor import SelfHealingMonitor
from self_healing_mlops.pipeline import evaluate_model, predict_file, train_model
from self_healing_mlops.retrain import maybe_retrain


def _write_sine(path: Path, freq: float, amplitude: float = 0.6, sr: int = 8000, duration: float = 0.2) -> None:
    frames = int(sr * duration)
    samples = []
    for n in range(frames):
        value = amplitude * math.sin(2 * math.pi * freq * n / sr)
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(int(v).to_bytes(2, byteorder="little", signed=True) for v in samples))


class TestAudioAndPipeline(unittest.TestCase):
    def test_extract_features_from_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "tone.wav"
            _write_sine(wav_path, freq=440)
            features = extract_features_from_wav(wav_path)

            self.assertGreater(features["duration"], 0)
            self.assertGreater(features["rms"], 0)
            self.assertGreaterEqual(features["zero_crossing_rate"], 0)

    def test_train_predict_evaluate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "real").mkdir()
            (base / "fake").mkdir()

            _write_sine(base / "real" / "r1.wav", freq=220, amplitude=0.2)
            _write_sine(base / "real" / "r2.wav", freq=240, amplitude=0.2)
            _write_sine(base / "fake" / "f1.wav", freq=1200, amplitude=0.8)
            _write_sine(base / "fake" / "f2.wav", freq=1000, amplitude=0.75)

            model_path = base / "model.json"
            train_result = train_model(base, model_path)
            self.assertEqual(train_result["samples"], 4)

            pred = predict_file(model_path, base / "fake" / "f1.wav")
            self.assertIn(pred["prediction"], (0, 1))

            eval_result = evaluate_model(base, model_path)
            self.assertEqual(eval_result["samples"], 4)
            self.assertGreaterEqual(eval_result["accuracy"], 0.5)

    def test_monitor_and_retrain_trigger(self) -> None:
        monitor = SelfHealingMonitor(accuracy_threshold=0.9, drift_threshold=0.1)
        called = {"value": False}

        def _callback() -> None:
            called["value"] = True

        triggered = maybe_retrain(
            monitor,
            recent_accuracy=0.5,
            reference_feature_mean=1.0,
            current_feature_mean=1.5,
            retrain_callback=_callback,
        )
        self.assertTrue(triggered)
        self.assertTrue(called["value"])


if __name__ == "__main__":
    unittest.main()
