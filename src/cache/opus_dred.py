from __future__ import annotations

import json
import subprocess
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf
import torch

from data import RawSpeechConfig, RawSpeechDataset
from loss_sim import PacketLossConfig, make_loss_mask


@dataclass(frozen=True)
class OpusDredCacheConfig:
    speech_path: str
    cache_dir: str
    opus_demo_path: str
    opus_root: str | None = None
    dred_checkpoint_dir: str | None = "checkpoints/dred"
    sample_rate: int = 16_000
    segment_seconds: float = 2.0
    split: str = "train"
    val_fraction: float = 0.02
    split_seed: int = 1234
    split_mode: str = "random"
    max_segments: int | None = None
    bitrate: int = 64_000
    dred_frames_10ms: int = 100
    random_loss_p: float = 0.05
    burst_loss_p: float = 0.02
    min_burst: int = 1
    max_burst: int = 5
    dry_run: bool = False
    show_progress: bool = True
    num_workers: int = 1



def _validate_opus_dred_runtime(config: OpusDredCacheConfig) -> dict[str, str]:
    opus_demo = Path(config.opus_demo_path)
    if not config.dry_run and not opus_demo.is_file():
        raise FileNotFoundError(
            f"opus_demo not found: {opus_demo}. Build Opus jstsp_dred with DRED enabled first: "
            "cd third_party/opus-jstsp-dred && ./autogen.sh && ./configure --enable-dred --enable-lossgen && make -j$(nproc)"
        )

    metadata: dict[str, str] = {"opus_demo_path": str(opus_demo)}
    if config.opus_root:
        opus_root = Path(config.opus_root)
        metadata["opus_root"] = str(opus_root)
        commit_file = Path(config.dred_checkpoint_dir or "") / "OPUS_JSTSP_DRED_COMMIT"
        if commit_file.is_file():
            metadata["expected_opus_commit"] = commit_file.read_text(encoding="utf-8").strip()
        head = opus_root / ".git"
        if head.exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(opus_root), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                metadata["opus_commit"] = result.stdout.strip()
            except subprocess.CalledProcessError:
                pass

    if config.dred_checkpoint_dir:
        ckpt_dir = Path(config.dred_checkpoint_dir)
        required = [
            "dred_rdovae.h",
            "dred_rdovae_enc.c",
            "dred_rdovae_enc.h",
            "dred_rdovae_dec.c",
            "dred_rdovae_dec.h",
        ]
        missing = [name for name in required if not (ckpt_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing Opus DRED C checkpoint artifacts in {ckpt_dir}: {missing}")
        metadata["dred_checkpoint_dir"] = str(ckpt_dir)
    return metadata



def _progress(iterable: Iterable, total: int, enabled: bool) -> Iterable:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="Building Opus DRED cache", unit="segment")


def _make_progress(total: int, enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="Building Opus DRED cache", unit="segment")


def _run_opus_demo_quiet(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    details = []
    if result.stdout:
        details.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr:
        details.append(f"stderr:\n{result.stderr.strip()}")
    suffix = "\n" + "\n".join(details) if details else ""
    raise RuntimeError(f"opus_demo failed with exit code {result.returncode}{suffix}")


def write_lossfile(path: str | Path, loss_mask: torch.Tensor) -> None:
    values = ["1" if bool(v) else "0" for v in loss_mask.flatten()]
    Path(path).write_text("\n".join(values) + "\n", encoding="utf-8")


def _write_pcm16(path: Path, audio: torch.Tensor) -> None:
    pcm = (audio.clamp(-1, 1) * 32767.0).round().to(torch.int16).cpu().numpy()
    pcm.tofile(path)


def _read_output_audio(path: Path, sample_rate: int, expected_samples: int) -> torch.Tensor:
    if path.suffix.lower() == ".wav":
        data, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            raise ValueError(f"expected sample rate {sample_rate}, got {sr}")
        audio = torch.as_tensor(data, dtype=torch.float32)
    else:
        import numpy as np

        audio = torch.from_numpy(np.fromfile(path, dtype="<i2").astype("float32")) / 32768.0
    if audio.dim() > 1:
        audio = audio[:, 0]
    if audio.numel() < expected_samples:
        audio = torch.nn.functional.pad(audio, (0, expected_samples - audio.numel()))
    return audio[:expected_samples]


def _build_cache_item(
    item: dict[str, Any],
    config: OpusDredCacheConfig,
    cache_dir: Path,
    items_dir: Path,
    work_dir: Path,
    frame_samples: int,
    loss_cfg: PacketLossConfig,
    runtime_metadata: dict[str, str],
) -> dict[str, Any]:
    audio = item["audio"].float()
    segment_index = int(item["segment_index"])
    n_frames = audio.numel() // frame_samples
    generator = torch.Generator().manual_seed(config.split_seed + segment_index)
    loss_mask = make_loss_mask(n_frames, loss_cfg, generator=generator)

    item_name = f"segment_{segment_index:09d}.pt"
    cache_path = items_dir / item_name
    input_pcm = work_dir / f"segment_{segment_index:09d}.pcm"
    lossfile = work_dir / f"segment_{segment_index:09d}.loss"
    output_pcm = work_dir / f"segment_{segment_index:09d}_dred.pcm"
    write_lossfile(lossfile, loss_mask)

    if config.dry_run:
        dred_audio = torch.zeros_like(audio)
        command = []
    else:
        _write_pcm16(input_pcm, audio)
        command = [
            config.opus_demo_path,
            "voip",
            str(config.sample_rate),
            "1",
            str(config.bitrate),
            "-lossfile",
            str(lossfile),
            "-dred",
            str(config.dred_frames_10ms),
            str(input_pcm),
            str(output_pcm),
        ]
        _run_opus_demo_quiet(command)
        dred_audio = _read_output_audio(output_pcm, config.sample_rate, audio.numel())

    payload = {
        "audio": audio,
        "dred_audio": dred_audio,
        "loss_mask": loss_mask,
        "segment_index": segment_index,
        "metadata": {
            "sample_rate": config.sample_rate,
            "segment_seconds": config.segment_seconds,
            "frame_samples": frame_samples,
            "dred_frames_10ms": config.dred_frames_10ms,
            "bitrate": config.bitrate,
            "opus_command": command,
            "dry_run": config.dry_run,
            "opus_dred_runtime": runtime_metadata,
        },
    }
    torch.save(payload, cache_path)
    return {
        "cache_path": str(cache_path.relative_to(cache_dir)),
        "segment_index": segment_index,
        "split": item["split"],
        "num_samples": int(audio.numel()),
        "num_frames": int(n_frames),
    }


def _write_manifest_record(manifest, record: dict[str, Any]) -> None:
    manifest.write(json.dumps(record, sort_keys=True) + "\n")
    manifest.flush()


def _iter_dataset_items(dataset: RawSpeechDataset) -> Iterable[dict[str, Any]]:
    for index in range(len(dataset)):
        yield dataset[index]


def _build_cache_parallel(
    dataset: RawSpeechDataset,
    config: OpusDredCacheConfig,
    cache_dir: Path,
    items_dir: Path,
    work_dir: Path,
    frame_samples: int,
    loss_cfg: PacketLossConfig,
    runtime_metadata: dict[str, str],
    manifest,
) -> None:
    max_pending = max(1, config.num_workers * 4)
    item_iter = iter(_iter_dataset_items(dataset))
    pending: set[Future[dict[str, Any]]] = set()
    progress = _make_progress(len(dataset), config.show_progress)

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            item = next(item_iter)
        except StopIteration:
            return False
        pending.add(
            executor.submit(
                _build_cache_item,
                item,
                config,
                cache_dir,
                items_dir,
                work_dir,
                frame_samples,
                loss_cfg,
                runtime_metadata,
            )
        )
        return True

    try:
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            for _ in range(min(max_pending, len(dataset))):
                submit_next(executor)

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    _write_manifest_record(manifest, future.result())
                    if progress is not None:
                        progress.update(1)
                    submit_next(executor)
    finally:
        if progress is not None:
            progress.close()


def build_opus_dred_cache(config: OpusDredCacheConfig) -> Path:
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    runtime_metadata = _validate_opus_dred_runtime(config)
    cache_dir = Path(config.cache_dir)
    items_dir = cache_dir / "items"
    work_dir = cache_dir / "work"
    items_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / f"manifest_{config.split}.jsonl"

    dataset = RawSpeechDataset(
        RawSpeechConfig(
            speech_path=config.speech_path,
            sample_rate=config.sample_rate,
            segment_seconds=config.segment_seconds,
            split=config.split,
            val_fraction=config.val_fraction,
            split_seed=config.split_seed,
            split_mode=config.split_mode,
            max_segments=config.max_segments,
        )
    )
    frame_samples = config.sample_rate // 50
    loss_cfg = PacketLossConfig(
        random_loss_p=config.random_loss_p,
        burst_loss_p=config.burst_loss_p,
        min_burst=config.min_burst,
        max_burst=config.max_burst,
    )

    with manifest_path.open("w", encoding="utf-8") as manifest:
        if config.num_workers == 1:
            for item in _progress(dataset, total=len(dataset), enabled=config.show_progress):
                record = _build_cache_item(
                    item,
                    config,
                    cache_dir,
                    items_dir,
                    work_dir,
                    frame_samples,
                    loss_cfg,
                    runtime_metadata,
                )
                _write_manifest_record(manifest, record)
        else:
            _build_cache_parallel(
                dataset,
                config,
                cache_dir,
                items_dir,
                work_dir,
                frame_samples,
                loss_cfg,
                runtime_metadata,
                manifest,
            )
    return manifest_path
