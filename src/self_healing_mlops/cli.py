from __future__ import annotations

import argparse
import json

from .monitor import SelfHealingMonitor
from .pipeline import evaluate_model, predict_file, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-healing MLOps deepfake voice detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from data directory")
    train_parser.add_argument("--data-dir", required=True)
    train_parser.add_argument("--model-path", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict class for a WAV file")
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--audio-file", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model against labeled folder")
    eval_parser.add_argument("--data-dir", required=True)
    eval_parser.add_argument("--model-path", required=True)

    monitor_parser = subparsers.add_parser("monitor", help="Check retraining condition")
    monitor_parser.add_argument("--recent-accuracy", type=float, required=True)
    monitor_parser.add_argument("--reference-mean", type=float, required=True)
    monitor_parser.add_argument("--current-mean", type=float, required=True)
    monitor_parser.add_argument("--accuracy-threshold", type=float, default=0.8)
    monitor_parser.add_argument("--drift-threshold", type=float, default=0.25)

    args = parser.parse_args()

    if args.command == "train":
        result = train_model(args.data_dir, args.model_path)
    elif args.command == "predict":
        result = predict_file(args.model_path, args.audio_file)
    elif args.command == "evaluate":
        result = evaluate_model(args.data_dir, args.model_path)
    else:
        monitor = SelfHealingMonitor(
            accuracy_threshold=args.accuracy_threshold,
            drift_threshold=args.drift_threshold,
        )
        monitor.update(args.recent_accuracy, args.reference_mean, args.current_mean)
        result = monitor.status()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
