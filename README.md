# codec-plc: Streaming Speech Codec with Packet-Loss Repair

`codec-plc` is a Python research project for a 16 kHz streaming speech codec that operates on 20 ms packets. It explores how decoder state and Opus DRED-derived context can be used to recover from lost speech packets in a low-latency streaming setting.

The repository includes a causal encoder/decoder, packet-loss simulation, offline Opus DRED cache generation, state/segment repair models, a DAC-native packet codec path, and scripts for data preparation, cache generation, training, validation, and real-time factor measurement.

## Core Ideas

- Audio is processed as a stream of 20 ms packets. At 16 kHz, each packet contains 320 samples.
- The base codec encodes and decodes short frames causally, without future context.
- Packet loss is modeled with random and burst-loss masks.
- Opus DRED is used as an external baseline and as training data for repair models.
- DRED outputs are cached offline so training does not invoke Opus for every batch.
- State repair and segment repair models learn to reconstruct useful decoder state or latents for missing packets.
- The DAC stage uses a local DAC-style architecture with one latent frame per 20 ms packet.

## Repository Layout

```text
src/codec          Base causal encoder/decoder, framing, quantizers
src/data           Raw 16 kHz int16 speech dataset support via np.memmap
src/cache          Offline Opus DRED cache builders and readers
src/dred           DRED provider interfaces and external adapter
src/state_repair   Mini-encoder and segment autoencoder for repair
src/loss_sim       Random and burst packet-loss simulation
src/dac            DAC-native packet codec, quantizer, losses, discriminator
src/eval           Real-time factor measurement helpers
scripts            Entrypoints for data preparation, cache, training, validation, RTF
configs            YAML experiment configs
```

Large local artifacts should stay out of git: `data/`, `cache/`, `runs/`, `checkpoints/`, `third_party/`, and local NISQA checkouts.

## Requirements

- Python 3.12.
- The project virtual environment, `.venv` (uv is recommended for creation).
- For real Opus DRED cache generation: a built `third_party/opus-jstsp-dred/opus_demo` binary with DRED support.
- For full dataset runs: `data/tts_speech_negative_16k.sw`, raw signed int16 little-endian mono PCM at 16 kHz.
- For NISQA-S validation: a local `NISQA-s/` checkout with config and weights.

The `.sw` speech file is taken from the Opus DRED repository: [https://media.xiph.org/lpcnet/speech/tts_speech_negative_16k.sw](https://media.xiph.org/lpcnet/speech/tts_speech_negative_16k.sw)

Create and activate the environment before running project commands:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Configuration-Only Runs

Runtime parameters are expected to live in YAML config files. Use `configs/default.yaml` as the base config, edit it for the run you want, or copy it to a separate file such as `configs/base_smoke.yaml`, `configs/repair.yaml`, or `configs/validation.yaml`.

The command line should only select the config file. Training does not enable ClearML logging by default.

## Data Preparation

The main dataset is expected as a raw `.sw` file containing 16 kHz int16 mono audio. `RawSpeechDataset` uses `np.memmap`, so the full file is not loaded into RAM.

Configure `dataset.speech_path`, `dataset.wav_dir`, `dataset.prepare_segment_seconds`, `dataset.prepare_train_fraction`, and `sample_rate`, then prepare WAV train/validation splits:

```bash
python scripts/prepare_sw.py --config configs/default.yaml
```

The script writes WAV files under `data/wav/train/` and `data/wav/val/` by default. Split behavior for training datasets is configured in `configs/default.yaml`, including `val_fraction`, `split_seed`, `split_mode`, and segment duration.

## Opus DRED Cache

For real repair-stage training, first build an offline cache containing Opus DRED outputs under deterministic packet-loss masks.

Configure `dataset`, `packet_loss`, and `dred_cache` in YAML, including `dred_cache.cache_dir`, `dred_cache.opus_root`, `dred_cache.opus_demo_path`, `dred_cache.dred_checkpoint_dir`, `dred_cache.dry_run`, `dred_cache.num_workers`, and `dataset.max_segments` for smoke-sized cache builds.

Build the cache selected by the config:

```bash
python scripts/build_dred_cache.py --config configs/default.yaml
```

Each cache item stores the original audio, Opus+DRED decoded audio, one loss-mask value per 20 ms packet, and Opus metadata. By default, manifests are written to `cache/dred/manifest_train.jsonl` and `cache/dred/manifest_val.jsonl`.

For code-only smoke checks without an external Opus binary, set `dred_cache.dry_run: true`. Do not use dry-run cache for real quality training.

## Training

Configure `training.stage` as `base`, `repair`, or `dac`, then run:

```bash
python scripts/train.py --config configs/default.yaml
```

Two-GPU training uses Accelerate for process launch and still keeps experiment parameters in YAML:

```bash
accelerate launch --config_file configs/accelerate_2gpu.yaml scripts/train.py --config configs/default.yaml
```

Supported stages:

- `base`: trains the base streaming speech codec.
- `repair`: trains segment/state repair on cached Opus DRED data.
- `dac`: trains the DAC-native packet codec with a frozen segment encoder for lost packets.

Important training settings live under `training`: stage, device, distributed mode, output directories, resume paths, epochs/steps, batch size, validation cadence, TensorBoard settings, dataloader workers, checkpoint paths, and logging. Cache manifests for repair and DAC stages live under `dred_cache.manifest_path` and `dred_cache.val_manifest_path`.

## Validation

Validation compares a trained codec against Opus DRED on the same validation segments and the same cached loss masks.

Set the `validation` section in YAML, especially `validation.checkpoint`, `validation.manifest`, `validation.output_dir`, `validation.max_segments`, and NISQA settings, then run:

```bash
python scripts/validate_model.py --config configs/default.yaml
```

Outputs are written to `metrics.json`, `metrics.csv`, `summary.md`, and, when audio saving is enabled, the `audio/` and `nisqa_48k/` subdirectories.

Primary metrics:

- PESQ-WB for full-reference comparison against the input audio.
- NISQA-S as a no-reference quality score.
- Raw packet payload bitrate for the codec path.
- RTF and chunks per second for speed measurement.
- Codec and repair-module parameter counts.

## Real-Time Factor

Configure the model in YAML, then measure encode/decode speed:

```bash
python scripts/eval_rtf.py --config configs/default.yaml
```

## Full Pipeline

A typical full run uses separate config files for cache, training, and validation so command lines stay stable:

```bash
source .venv/bin/activate
```

```bash
cd third_party/opus-jstsp-dred
./autogen.sh
./configure --enable-dred --enable-lossgen
make -j"$(nproc)"
cd ../..
```

```bash
python scripts/build_dred_cache.py --config configs/default.yaml
```

```bash
accelerate launch --config_file configs/accelerate_2gpu.yaml scripts/train.py --config configs/default.yaml
```

```bash
python scripts/validate_model.py --config configs/default.yaml
```

## Practical Notes

- Keep `dred_cache.opus_root`, `dred_cache.opus_demo_path`, and `dred_cache.dred_checkpoint_dir` configurable.
- The current Opus DRED checkpoint is C weights under `checkpoints/dred`, not a PyTorch `.pth` checkpoint.
- Under `accelerate launch`, `training.batch_size` is per process; on two GPUs, the global batch size is doubled.
- Until a packet serializer and entropy coder exist, the raw codec bitrate is not a final transport bitrate.
