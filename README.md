# Self-healing-mlops-for-Deepfake-voice-detection

A baseline implementation for deepfake voice detection with a simple self-healing MLOps loop.

## What is implemented

- WAV audio ingestion (16-bit PCM) and feature extraction
- Baseline deepfake-vs-real classifier (prototype distance model)
- Training, evaluation, and single-file prediction pipeline
- Self-healing monitor for accuracy drop and feature drift
- CLI for train/predict/evaluate/monitor flows
- Unit tests for core behavior

## Project structure

- `src/self_healing_mlops/audio.py` — audio reading and feature extraction
- `src/self_healing_mlops/model.py` — baseline classifier
- `src/self_healing_mlops/pipeline.py` — train/evaluate/predict utilities
- `src/self_healing_mlops/monitor.py` — drift + performance monitor
- `src/self_healing_mlops/retrain.py` — retraining trigger callback helper
- `src/self_healing_mlops/cli.py` — command line interface
- `tests/test_audio_and_model.py` — targeted tests

## Quick start

From the repository root:

```bash
python -m pip install -e .
```

Dataset layout expected for training/evaluation:

```text
data/
  real/
    *.wav
  fake/
    *.wav
```

Train model:

```bash
shmlops train --data-dir data --model-path model.json
```

Predict one file:

```bash
shmlops predict --model-path model.json --audio-file sample.wav
```

Evaluate model:

```bash
shmlops evaluate --data-dir data --model-path model.json
```

Monitor health (self-healing trigger condition):

```bash
shmlops monitor --recent-accuracy 0.72 --reference-mean 0.19 --current-mean 0.29
```

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```
