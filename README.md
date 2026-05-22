# IMSIT Streaming Codec Scaffold

This repo contains a first research implementation for a 16 kHz, 20 ms streaming speech
codec with DRED-based decoder-state repair.

Implemented pieces:

- causal encoder/decoder with explicit `DecoderState`,
- residual scalar/vector quantizer with variable active stages,
- packet-loss simulator with random and burst losses,
- DRED provider interface plus external Opus/RDOVAE command adapter,
- mini-encoder that predicts additive decoder-state deltas from DRED audio,
- smoke training and RTF evaluation scripts.

The Opus DRED/RDOVAE code is intentionally not vendored. Use `ExternalDredProvider` to point
at a local checkout or wrapper for:

```bash
<cmd> --input in.wav --start-sample N --num-samples M --output out.wav
```

Quick checks:

```bash
source .venv/bin/activate
pytest -q
python scripts/train_smoke.py
python scripts/eval_rtf.py
```

Real-data workflow:

```bash
source .venv/bin/activate
python scripts/build_dred_cache.py --config configs/default.yaml --split train --max-segments 8 --dry-run --workers 4
python scripts/build_dred_cache.py --config configs/default.yaml --split val --max-segments 8 --dry-run --workers 4
python scripts/train.py --config configs/default.yaml --stage base --epochs 1 --output-dir runs/base_smoke --max-segments 256 --max-val-segments 64 --val-every-epochs 1 --no-clearml
python scripts/train.py --config configs/default.yaml --stage repair --epochs 1 --output-dir runs/repair_smoke --val-every-epochs 1 --no-clearml
```

For non-dry-run DRED cache generation, set `dred_cache.opus_demo_path` to an Opus binary built from the external `jstsp_dred` checkout with DRED enabled. Normal training runtime arguments live in `configs/default.yaml`; CLI flags are overrides. Use `--workers 8` for parallel cache generation on a multi-core CPU; lower it if disk or CPU is saturated.
