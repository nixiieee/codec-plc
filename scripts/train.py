from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamloss_codec.cache.dred_cache import CachedDredDataset, collate_cached_dred  # noqa: E402
from streamloss_codec.codec import StreamingSpeechCodec  # noqa: E402
from streamloss_codec.config import load_config  # noqa: E402
from streamloss_codec.data import RawSpeechConfig, RawSpeechDataset  # noqa: E402
from streamloss_codec.state_repair import StateRepairMiniEncoder  # noqa: E402
from streamloss_codec.train import (  # noqa: E402
    evaluate_base_batch,
    evaluate_cached_repair_batch,
    train_base_step,
    train_cached_repair_step,
)


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg["model"].items() if k != "active_quantizers_train"}


def _cfg_value(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    return cfg.get("training", {}).get(key, default)


def _stage_value(cfg: dict[str, Any], stage: str, key: str, default: Any = None) -> Any:
    training_cfg = cfg.get("training", {})
    return training_cfg.get(f"{stage}_{key}", training_cfg.get(key, default))


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _apply_config_defaults(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    stage = _coalesce(args.stage, _cfg_value(cfg, "stage", "base"))
    if stage not in {"base", "repair"}:
        raise ValueError(f"stage must be base or repair, got {stage!r}")
    args.stage = stage

    args.output_dir = _coalesce(args.output_dir, _stage_value(cfg, stage, "output_dir"), "runs/default")
    args.resume = _coalesce(args.resume, _stage_value(cfg, stage, "resume"))
    args.cache_manifest = _coalesce(args.cache_manifest, cfg.get("dred_cache", {}).get("manifest_path"))
    args.val_cache_manifest = _coalesce(args.val_cache_manifest, cfg.get("dred_cache", {}).get("val_manifest_path"))
    args.log_file = _coalesce(args.log_file, _cfg_value(cfg, "log_file"))
    args.log_every = int(_coalesce(args.log_every, _cfg_value(cfg, "log_every", 10)))
    args.save_every = int(_coalesce(args.save_every, _cfg_value(cfg, "save_every", 100)))
    args.val_every_epochs = int(_coalesce(args.val_every_epochs, _cfg_value(cfg, "val_every_epochs", 0)))
    args.val_max_batches = _coalesce(args.val_max_batches, _cfg_value(cfg, "val_max_batches"))
    args.device = _coalesce(args.device, _cfg_value(cfg, "device", "auto"))
    args.batch_size = _coalesce(args.batch_size, _cfg_value(cfg, "batch_size"))
    args.num_workers = int(_coalesce(args.num_workers, _cfg_value(cfg, "num_workers", 0)))
    args.max_segments = _coalesce(args.max_segments, _cfg_value(cfg, "max_segments"), cfg.get("dataset", {}).get("max_segments"))
    args.max_val_segments = _coalesce(args.max_val_segments, _cfg_value(cfg, "max_val_segments"))
    args.distributed = bool(_coalesce(args.distributed, _cfg_value(cfg, "distributed", False)))
    args.no_progress = not bool(_coalesce(args.progress, _cfg_value(cfg, "progress", True)))
    args.no_clearml = not bool(_coalesce(args.clearml, _cfg_value(cfg, "clearml_enabled", True)))
    args.clearml_project = _coalesce(args.clearml_project, _cfg_value(cfg, "clearml_project", "imsit"))
    args.clearml_task_name = _coalesce(args.clearml_task_name, _cfg_value(cfg, "clearml_task_name"))
    args.epochs = _coalesce(args.epochs, _stage_value(cfg, stage, "epochs"))
    args.steps = _coalesce(args.steps, _stage_value(cfg, stage, "steps"))
    return args


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _make_accelerator(args: argparse.Namespace):
    use_accelerate = args.distributed or _world_size() > 1
    if not use_accelerate:
        return None
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("Distributed training requires accelerate. Install it with: uv add accelerate") from exc
    return Accelerator()


def _is_main(accelerator: object | None) -> bool:
    return accelerator is None or accelerator.is_main_process


def _device(args: argparse.Namespace, accelerator: object | None) -> torch.device:
    if accelerator is not None:
        return accelerator.device
    return _resolve_device(args.device)


def _save_checkpoint(path: Path, payload: dict[str, Any], accelerator: object | None = None) -> None:
    if not _is_main(accelerator):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _state_dict(module: torch.nn.Module, accelerator: object | None = None) -> dict[str, torch.Tensor]:
    if accelerator is not None:
        module = accelerator.unwrap_model(module)
    state_dict = module.state_dict()
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.replace("decoder.module.", "decoder.")
        cleaned[clean_key] = value.detach().cpu()
    return cleaned


def _load_checkpoint(path: str | None, codec: StreamingSpeechCodec, repair: StateRepairMiniEncoder | None = None) -> int:
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    codec.load_state_dict(checkpoint["codec_state_dict"], strict=False)
    if repair is not None and "repair_state_dict" in checkpoint:
        repair.load_state_dict(checkpoint["repair_state_dict"], strict=False)
    return int(checkpoint.get("step", 0))


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


class TrainLogger:
    def __init__(self, path: Path | None, accelerator: object | None = None) -> None:
        self.path = path
        self.handle = None
        if path is not None and _is_main(accelerator):
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a", encoding="utf-8")

    def log(self, payload: dict[str, Any]) -> None:
        if self.handle is None:
            return
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class ClearMLLogger:
    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any], stage: str, accelerator: object | None) -> None:
        self.task = None
        self.logger = None
        if args.no_clearml or not _is_main(accelerator):
            return
        try:
            from clearml import Task
        except ImportError:
            return
        task_name = args.clearml_task_name or f"{stage}:{Path(args.output_dir).name}"
        self.task = Task.init(project_name=args.clearml_project, task_name=task_name, reuse_last_task_id=False)
        self.task.connect(cfg, name="config")
        self.task.connect(vars(args), name="args")
        self.logger = self.task.get_logger()

    def report(self, split: str, metrics: dict[str, float], step: int, epoch: int) -> None:
        if self.logger is None:
            return
        for name, value in metrics.items():
            self.logger.report_scalar(title=name, series=split, value=float(value), iteration=step)
        self.logger.report_scalar(title="epoch", series=split, value=float(epoch), iteration=step)

    def close(self) -> None:
        if self.task is not None:
            self.task.close()
            self.task = None
            self.logger = None

    def __enter__(self) -> "ClearMLLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _log_file_path(args: argparse.Namespace, stage: str) -> Path:
    if args.log_file is not None:
        return Path(args.log_file)
    return Path(args.output_dir) / f"{stage}.log"


def _make_epoch_progress(*, stage: str, epoch: int, step: int, target_steps: int, loader_len: int, accelerator: object | None, enabled: bool):
    if not enabled or not _is_main(accelerator):
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    total = min(loader_len, max(0, target_steps - step))
    if total <= 0:
        return None
    return tqdm(total=total, desc=f"{stage} epoch {epoch}", unit="batch", dynamic_ncols=True)


def _progress_set_postfix(progress, values: dict[str, float]) -> None:
    if progress is None:
        return
    progress.set_postfix({key: f"{value:.4f}" for key, value in values.items()})


def _make_raw_dataset(cfg: dict[str, Any], split: str, max_segments: int | None) -> RawSpeechDataset:
    dataset_cfg = cfg["dataset"]
    return RawSpeechDataset(
        RawSpeechConfig(
            speech_path=dataset_cfg["speech_path"],
            sample_rate=cfg["sample_rate"],
            segment_seconds=dataset_cfg["segment_seconds"],
            split=split,
            val_fraction=dataset_cfg.get("val_fraction", 0.02),
            split_seed=dataset_cfg.get("split_seed", 1234),
            max_segments=max_segments,
        )
    )


def _configured_epochs(args: argparse.Namespace, cfg: dict[str, Any], stage: str) -> int | None:
    if args.epochs is not None:
        return args.epochs
    return cfg["training"].get("base_epochs" if stage == "base" else "repair_epochs")


def _target_steps(args: argparse.Namespace, cfg: dict[str, Any], stage: str, train_loader_len: int) -> int:
    if args.steps is not None:
        return args.steps
    epochs = _configured_epochs(args, cfg, stage)
    if epochs is not None:
        return int(epochs) * train_loader_len
    steps = cfg["training"].get("base_steps" if stage == "base" else "repair_steps")
    if steps is None:
        raise ValueError(f"training.{stage}_epochs or training.{stage}_steps must be set")
    return int(steps)


def _target_epochs(args: argparse.Namespace, cfg: dict[str, Any], stage: str, target_steps: int, train_loader_len: int) -> int:
    epochs = _configured_epochs(args, cfg, stage)
    if epochs is not None:
        return int(epochs)
    return (target_steps + train_loader_len - 1) // train_loader_len


def _infer_val_cache_manifest(train_manifest: str | None) -> str | None:
    if not train_manifest:
        return None
    path = Path(train_manifest)
    candidates = []
    if "train" in path.name:
        candidates.append(path.with_name(path.name.replace("train", "val")))
    candidates.append(path.with_name("manifest_val.jsonl"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _prepare_base(codec, optimizer, train_loader, val_loader, accelerator):
    if accelerator is None:
        return codec, optimizer, train_loader, val_loader
    if val_loader is None:
        codec, optimizer, train_loader = accelerator.prepare(codec, optimizer, train_loader)
        return codec, optimizer, train_loader, None
    return accelerator.prepare(codec, optimizer, train_loader, val_loader)


def _prepare_repair(codec, repair, optimizer, train_loader, val_loader, accelerator):
    if accelerator is None:
        return codec, repair, optimizer, train_loader, val_loader
    codec.encoder.to(accelerator.device)
    codec.quantizer.to(accelerator.device)
    if val_loader is None:
        decoder, repair, optimizer, train_loader = accelerator.prepare(codec.decoder, repair, optimizer, train_loader)
        codec.decoder = decoder
        return codec, repair, optimizer, train_loader, None
    decoder, repair, optimizer, train_loader, val_loader = accelerator.prepare(codec.decoder, repair, optimizer, train_loader, val_loader)
    codec.decoder = decoder
    return codec, repair, optimizer, train_loader, val_loader


def _losses_to_vector(losses, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(losses.total) * batch_size,
            float(losses.reconstruction) * batch_size,
            float(losses.quantizer) * batch_size,
            float(losses.repair) * batch_size,
            float(batch_size),
        ],
        dtype=torch.float32,
        device=device,
    )


def _reduce_metrics(total: torch.Tensor, accelerator: object | None) -> dict[str, float]:
    if accelerator is not None:
        gathered = accelerator.gather(total.unsqueeze(0))
        total = gathered.sum(dim=0)
    count = max(float(total[4].item()), 1.0)
    return {
        "total": float((total[0] / count).item()),
        "recon": float((total[1] / count).item()),
        "quantizer": float((total[2] / count).item()),
        "repair": float((total[3] / count).item()),
    }


def validate_base(codec, val_loader, active_quantizers: int, device: torch.device, accelerator: object | None, max_batches: int | None) -> dict[str, float] | None:
    if val_loader is None:
        return None
    total = torch.zeros(5, dtype=torch.float32, device=device)
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        audio = batch["audio"].float().to(device, non_blocking=True)
        losses = evaluate_base_batch(codec, audio, active_quantizers=active_quantizers)
        total += _losses_to_vector(losses, audio.shape[0], device)
    return _reduce_metrics(total, accelerator)


def validate_repair(codec, repair, val_loader, active_quantizers: int, device: torch.device, cfg: dict[str, Any], accelerator: object | None, max_batches: int | None) -> dict[str, float] | None:
    if val_loader is None:
        return None
    total = torch.zeros(5, dtype=torch.float32, device=device)
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        audio = batch["audio"].float().to(device, non_blocking=True)
        losses = evaluate_cached_repair_batch(
            codec,
            repair,
            audio,
            batch["dred_audio"].float().to(device, non_blocking=True),
            batch["loss_mask"].bool().to(device, non_blocking=True),
            active_quantizers=active_quantizers,
            sample_rate=cfg["sample_rate"],
            chunk_ms=cfg["chunk_ms"],
        )
        total += _losses_to_vector(losses, audio.shape[0], device)
    return _reduce_metrics(total, accelerator)


def _should_validate(args: argparse.Namespace, epoch: int, final_epoch: bool) -> bool:
    if args.val_every_epochs <= 0:
        return False
    return epoch % args.val_every_epochs == 0 or final_epoch


def run_base(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    accelerator = _make_accelerator(args)
    device = _device(args, accelerator)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = _make_raw_dataset(cfg, cfg["dataset"].get("split", "train"), args.max_segments if args.max_segments is not None else cfg["dataset"].get("max_segments"))
    val_dataset = _make_raw_dataset(cfg, "val", args.max_val_segments) if args.val_every_epochs > 0 else None
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    codec = StreamingSpeechCodec(**_model_kwargs(cfg))
    start_step = _load_checkpoint(args.resume, codec)
    if accelerator is None:
        codec.to(device)
    optimizer = torch.optim.Adam(codec.parameters(), lr=cfg["training"]["base_lr"])
    active = cfg["model"].get("active_quantizers_train", [codec.quantizer.num_quantizers])
    codec, optimizer, train_loader, val_loader = _prepare_base(codec, optimizer, train_loader, val_loader, accelerator)

    out_dir = Path(args.output_dir)
    log_path = _log_file_path(args, "base")
    target_steps = _target_steps(args, cfg, "base", len(train_loader))
    target_epochs = _target_epochs(args, cfg, "base", target_steps, len(train_loader))

    with TrainLogger(log_path, accelerator) as logger, ClearMLLogger(args, cfg, "base", accelerator) as clearml_logger:
        logger.log({
            "event": "start",
            "stage": "base",
            "device": str(device),
            "distributed": accelerator is not None,
            "num_processes": getattr(accelerator, "num_processes", 1),
            "dataset_segments": len(train_dataset),
            "val_segments": 0 if val_dataset is None else len(val_dataset),
            "loader_batches_per_process": len(train_loader),
            "batch_size_per_process": batch_size,
            "start_step": start_step,
            "target_steps": target_steps,
            "target_epochs": target_epochs,
            "log_file": str(log_path),
        })

        step = start_step
        epoch = step // max(1, len(train_loader))
        while step < target_steps:
            epoch += 1
            progress = _make_epoch_progress(stage="base", epoch=epoch, step=step, target_steps=target_steps, loader_len=len(train_loader), accelerator=accelerator, enabled=not args.no_progress)
            try:
                for batch in train_loader:
                    codec.train()
                    audio = batch["audio"].float().to(device, non_blocking=True)
                    active_quantizers = active[step % len(active)]
                    losses = train_base_step(codec, optimizer, audio, active_quantizers, accelerator=accelerator)
                    step += 1
                    metrics = {"total": float(losses.total), "recon": float(losses.reconstruction), "quantizer": float(losses.quantizer)}
                    if progress is not None:
                        progress.update(1)
                        _progress_set_postfix(progress, {"loss": metrics["total"], "recon": metrics["recon"]})
                    if step % args.log_every == 0:
                        logger.log({"event": "metrics", "split": "train", "stage": "base", "epoch": epoch, "step": step, "active_quantizers": active_quantizers, **metrics})
                        clearml_logger.report("train", metrics, step, epoch)
                    if step % args.save_every == 0 or step >= target_steps:
                        checkpoint_path = out_dir / f"base_step_{step}.pt"
                        _save_checkpoint(checkpoint_path, {"step": step, "epoch": epoch, "codec_state_dict": _state_dict(codec, accelerator), "config": cfg}, accelerator)
                        logger.log({"event": "checkpoint", "stage": "base", "epoch": epoch, "step": step, "path": str(checkpoint_path)})
                    if step >= target_steps:
                        break
            finally:
                if progress is not None:
                    progress.close()

            if _should_validate(args, epoch, step >= target_steps):
                val_metrics = validate_base(codec, val_loader, active[-1], device, accelerator, args.val_max_batches)
                if val_metrics is not None:
                    logger.log({"event": "validation", "split": "val", "stage": "base", "epoch": epoch, "step": step, **val_metrics})
                    clearml_logger.report("val", val_metrics, step, epoch)
        logger.log({"event": "finish", "stage": "base", "step": step, "epoch": epoch})
    if accelerator is not None:
        accelerator.wait_for_everyone()
        accelerator.end_training()


def run_repair(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    accelerator = _make_accelerator(args)
    device = _device(args, accelerator)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    manifest = args.cache_manifest or cfg["dred_cache"].get("manifest_path")
    if not manifest:
        raise ValueError("repair stage requires --cache-manifest or dred_cache.manifest_path")
    val_manifest = args.val_cache_manifest or _infer_val_cache_manifest(manifest)
    if args.val_every_epochs > 0 and not val_manifest:
        raise ValueError("repair validation requires --val-cache-manifest or a manifest_val.jsonl next to the train manifest")

    train_dataset = CachedDredDataset(manifest)
    val_dataset = CachedDredDataset(val_manifest) if val_manifest and args.val_every_epochs > 0 else None
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_cached_dred)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_cached_dred)

    codec = StreamingSpeechCodec(**_model_kwargs(cfg))
    repair = StateRepairMiniEncoder(decoder_hidden=cfg["model"]["decoder_hidden"])
    start_step = _load_checkpoint(args.resume, codec, repair)
    if accelerator is None:
        codec.to(device)
        repair.to(device)
    _set_requires_grad(codec.encoder, False)
    _set_requires_grad(codec.quantizer, False)
    _set_requires_grad(codec.decoder, True)
    optimizer = torch.optim.Adam(list(codec.decoder.parameters()) + list(repair.parameters()), lr=cfg["training"]["repair_lr"])
    active = cfg["model"].get("active_quantizers_train", [codec.quantizer.num_quantizers])
    codec, repair, optimizer, train_loader, val_loader = _prepare_repair(codec, repair, optimizer, train_loader, val_loader, accelerator)

    out_dir = Path(args.output_dir)
    log_path = _log_file_path(args, "repair")
    target_steps = _target_steps(args, cfg, "repair", len(train_loader))
    target_epochs = _target_epochs(args, cfg, "repair", target_steps, len(train_loader))

    with TrainLogger(log_path, accelerator) as logger, ClearMLLogger(args, cfg, "repair", accelerator) as clearml_logger:
        logger.log({
            "event": "start",
            "stage": "repair",
            "device": str(device),
            "distributed": accelerator is not None,
            "num_processes": getattr(accelerator, "num_processes", 1),
            "dataset_segments": len(train_dataset),
            "val_segments": 0 if val_dataset is None else len(val_dataset),
            "loader_batches_per_process": len(train_loader),
            "batch_size_per_process": batch_size,
            "start_step": start_step,
            "target_steps": target_steps,
            "target_epochs": target_epochs,
            "cache_manifest": str(manifest),
            "val_cache_manifest": None if val_manifest is None else str(val_manifest),
            "log_file": str(log_path),
        })

        step = start_step
        epoch = step // max(1, len(train_loader))
        while step < target_steps:
            epoch += 1
            progress = _make_epoch_progress(stage="repair", epoch=epoch, step=step, target_steps=target_steps, loader_len=len(train_loader), accelerator=accelerator, enabled=not args.no_progress)
            try:
                for batch in train_loader:
                    codec.train()
                    repair.train()
                    active_quantizers = active[step % len(active)]
                    losses = train_cached_repair_step(
                        codec,
                        repair,
                        optimizer,
                        batch["audio"].float().to(device, non_blocking=True),
                        batch["dred_audio"].float().to(device, non_blocking=True),
                        batch["loss_mask"].bool().to(device, non_blocking=True),
                        active_quantizers=active_quantizers,
                        sample_rate=cfg["sample_rate"],
                        chunk_ms=cfg["chunk_ms"],
                        accelerator=accelerator,
                    )
                    step += 1
                    metrics = {"total": float(losses.total), "recon": float(losses.reconstruction), "repair": float(losses.repair)}
                    if progress is not None:
                        progress.update(1)
                        _progress_set_postfix(progress, {"loss": metrics["total"], "repair": metrics["repair"]})
                    if step % args.log_every == 0:
                        logger.log({"event": "metrics", "split": "train", "stage": "repair", "epoch": epoch, "step": step, "active_quantizers": active_quantizers, **metrics})
                        clearml_logger.report("train", metrics, step, epoch)
                    if step % args.save_every == 0 or step >= target_steps:
                        checkpoint_path = out_dir / f"repair_step_{step}.pt"
                        _save_checkpoint(checkpoint_path, {"step": step, "epoch": epoch, "codec_state_dict": _state_dict(codec, accelerator), "repair_state_dict": _state_dict(repair, accelerator), "config": cfg}, accelerator)
                        logger.log({"event": "checkpoint", "stage": "repair", "epoch": epoch, "step": step, "path": str(checkpoint_path)})
                    if step >= target_steps:
                        break
            finally:
                if progress is not None:
                    progress.close()

            if _should_validate(args, epoch, step >= target_steps):
                val_metrics = validate_repair(codec, repair, val_loader, active[-1], device, cfg, accelerator, args.val_max_batches)
                if val_metrics is not None:
                    logger.log({"event": "validation", "split": "val", "stage": "repair", "epoch": epoch, "step": step, **val_metrics})
                    clearml_logger.report("val", val_metrics, step, epoch)
        logger.log({"event": "finish", "stage": "repair", "step": step, "epoch": epoch})
    if accelerator is not None:
        accelerator.wait_for_everyone()
        accelerator.end_training()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train streaming codec stages.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage", choices=["base", "repair"], default=None, help="Override training.stage")
    parser.add_argument("--steps", type=int, default=None, help="Override training length in optimizer steps")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count")
    parser.add_argument("--output-dir", default=None, help="Override configured output_dir")
    parser.add_argument("--resume", default=None, help="Override configured resume checkpoint")
    parser.add_argument("--cache-manifest", default=None, help="Override dred_cache.manifest_path")
    parser.add_argument("--val-cache-manifest", default=None, help="Override dred_cache.val_manifest_path")
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--log-file", default=None, help="Write JSONL training logs here; defaults to <output-dir>/<stage>.log")
    parser.add_argument("--progress", dest="progress", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable tqdm progress bars")
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--val-every-epochs", type=int, default=None, help="Run validation every N epochs; 0 disables validation")
    parser.add_argument("--val-max-batches", type=int, default=None, help="Limit validation batches for smoke checks")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...; ignored under accelerate launch")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size per process")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers per process")
    parser.add_argument("--max-segments", type=int, default=None, help="Override dataset.max_segments for base stage")
    parser.add_argument("--max-val-segments", type=int, default=None, help="Limit base validation segments")
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=None, help="Force Accelerate mode; accelerate launch sets this automatically via WORLD_SIZE")
    parser.add_argument("--clearml", dest="clearml", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable ClearML logging")
    parser.add_argument("--clearml-project", default=None, help="ClearML project name")
    parser.add_argument("--clearml-task-name", default=None, help="ClearML task name; defaults to <stage>:<output-dir name>")
    args = parser.parse_args()
    cfg = load_config(args.config)
    args = _apply_config_defaults(args, cfg)
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.val_every_epochs < 0:
        raise ValueError("val_every_epochs must be >= 0")
    if args.stage == "base":
        run_base(cfg, args)
    else:
        run_repair(cfg, args)


if __name__ == "__main__":
    main()
