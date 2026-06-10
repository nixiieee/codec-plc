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
from streamloss_codec.state_repair import SegmentRepairAutoencoder  # noqa: E402
from streamloss_codec.train import (  # noqa: E402
    evaluate_base_batch,
    evaluate_segment_repair_ae_batch,
    reconstruct_lost_segments_with_ae,
    train_base_step,
    train_segment_repair_ae_step,
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
    args.tensorboard = bool(_coalesce(args.tensorboard, _cfg_value(cfg, "tensorboard_enabled", True)))
    args.tensorboard_dir = _coalesce(args.tensorboard_dir, _cfg_value(cfg, "tensorboard_dir"))
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


def _unwrap_module(module: torch.nn.Module, accelerator: object | None = None) -> torch.nn.Module:
    if accelerator is not None:
        return accelerator.unwrap_model(module)
    return module.module if hasattr(module, "module") else module


def _state_dict(module: torch.nn.Module, accelerator: object | None = None) -> dict[str, torch.Tensor]:
    module = _unwrap_module(module, accelerator)
    state_dict = module.state_dict()
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key.replace("decoder.module.", "decoder.")
        cleaned[clean_key] = value.detach().cpu()
    return cleaned


def _load_checkpoint(path: str | None, codec: StreamingSpeechCodec) -> int:
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    codec.load_state_dict(checkpoint["codec_state_dict"], strict=False)
    return int(checkpoint.get("step", 0))


def _load_segment_ae_checkpoint(path: str | None, model: SegmentRepairAutoencoder, optimizer: torch.optim.Optimizer | None = None) -> int:
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "segment_ae_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["segment_ae_state_dict"], strict=False)
    elif "segment_encoder_state_dict" in checkpoint:
        model.encoder.load_state_dict(checkpoint["segment_encoder_state_dict"], strict=False)
    else:
        raise KeyError("checkpoint must contain segment_ae_state_dict or segment_encoder_state_dict")
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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


class TensorBoardLogger:
    def __init__(self, args: argparse.Namespace, stage: str, accelerator: object | None) -> None:
        self.writer = None
        if not args.tensorboard or not _is_main(accelerator):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return
        log_dir = Path(args.tensorboard_dir) if args.tensorboard_dir is not None else Path(args.output_dir) / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(log_dir / stage))

    def report(self, split: str, metrics: dict[str, float | None], step: int, epoch: int) -> None:
        if self.writer is None:
            return
        for name, value in metrics.items():
            if value is None:
                continue
            self.writer.add_scalar(f"{split}/{name}", float(value), step)
        self.writer.add_scalar(f"{split}/epoch", float(epoch), step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "TensorBoardLogger":
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


def _prepare_repair(model, optimizer, train_loader, val_loader, accelerator):
    if accelerator is None:
        return model, optimizer, train_loader, val_loader
    if val_loader is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        return model, optimizer, train_loader, None
    return accelerator.prepare(model, optimizer, train_loader, val_loader)


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


def _pesq_score(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int) -> float | None:
    try:
        from pesq import pesq
    except ImportError:
        return None
    if sample_rate != 16_000:
        return None
    try:
        return float(pesq(sample_rate, reference.detach().cpu().numpy(), degraded.detach().cpu().numpy(), "wb"))
    except Exception:
        return None


def _segment_losses_to_vector(losses, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(losses.total) * losses.lost_frames,
            float(losses.mse) * losses.lost_frames,
            float(losses.l1) * losses.lost_frames,
            float(losses.stft) * losses.lost_frames,
            float(losses.lost_frames),
        ],
        dtype=torch.float32,
        device=device,
    )


def _reduce_segment_metrics(total: torch.Tensor, accelerator: object | None) -> dict[str, float]:
    if accelerator is not None:
        gathered = accelerator.gather(total.unsqueeze(0))
        total = gathered.sum(dim=0)
    count = max(float(total[4].item()), 1.0)
    return {
        "total": float((total[0] / count).item()),
        "mse": float((total[1] / count).item()),
        "l1": float((total[2] / count).item()),
        "stft": float((total[3] / count).item()),
        "lost_frames": float(total[4].item()),
    }


def validate_repair(model, val_loader, device: torch.device, cfg: dict[str, Any], accelerator: object | None, max_batches: int | None) -> dict[str, float | None] | None:
    if val_loader is None:
        return None
    total = torch.zeros(5, dtype=torch.float32, device=device)
    pesq_values = []
    dred_pesq_values = []
    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        audio = batch["audio"].float().to(device, non_blocking=True)
        dred_audio = batch["dred_audio"].float().to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].bool().to(device, non_blocking=True)
        losses = evaluate_segment_repair_ae_batch(
            model,
            audio,
            dred_audio,
            loss_mask,
            sample_rate=cfg["sample_rate"],
            chunk_ms=cfg["chunk_ms"],
        )
        total += _segment_losses_to_vector(losses, device)
        if _is_main(accelerator):
            patched = reconstruct_lost_segments_with_ae(
                model,
                dred_audio,
                loss_mask,
                sample_rate=cfg["sample_rate"],
                chunk_ms=cfg["chunk_ms"],
            )
            for ref, pred, dred in zip(audio, patched, dred_audio, strict=False):
                score = _pesq_score(ref, pred[: ref.numel()], cfg["sample_rate"])
                if score is not None:
                    pesq_values.append(score)
                dred_score = _pesq_score(ref, dred[: ref.numel()], cfg["sample_rate"])
                if dred_score is not None:
                    dred_pesq_values.append(dred_score)
    metrics: dict[str, float | None] = _reduce_segment_metrics(total, accelerator)
    if _is_main(accelerator):
        metrics["pesq_wb"] = sum(pesq_values) / len(pesq_values) if pesq_values else None
        metrics["pesq_wb_dred_baseline"] = sum(dred_pesq_values) / len(dred_pesq_values) if dred_pesq_values else None
    else:
        metrics["pesq_wb"] = None
        metrics["pesq_wb_dred_baseline"] = None
    return metrics


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

    with TrainLogger(log_path, accelerator) as logger, TensorBoardLogger(args, "base", accelerator) as tb_logger:
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
                        tb_logger.report("train", metrics, step, epoch)
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
                    tb_logger.report("val", val_metrics, step, epoch)
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

    repair_cfg = cfg.get("segment_repair", {})
    model = SegmentRepairAutoencoder(
        channels=int(repair_cfg.get("channels", cfg["model"].get("channels", 136))),
        latent_dim=int(repair_cfg.get("latent_dim", 96)),
        latent_frames=int(repair_cfg.get("latent_frames", 8)),
        residual=bool(repair_cfg.get("residual", True)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["repair_lr"])
    start_step = _load_segment_ae_checkpoint(args.resume, model, optimizer)
    if accelerator is None:
        model.to(device)
    model, optimizer, train_loader, val_loader = _prepare_repair(model, optimizer, train_loader, val_loader, accelerator)

    out_dir = Path(args.output_dir)
    log_path = _log_file_path(args, "repair")
    target_steps = _target_steps(args, cfg, "repair", len(train_loader))
    target_epochs = _target_epochs(args, cfg, "repair", target_steps, len(train_loader))

    with TrainLogger(log_path, accelerator) as logger, TensorBoardLogger(args, "repair", accelerator) as tb_logger:
        logger.log({
            "event": "start",
            "stage": "repair",
            "mode": "segment_ae_pretrain",
            "device": str(device),
            "distributed": accelerator is not None,
            "num_processes": getattr(accelerator, "num_processes", 1),
            "dataset_segments": len(train_dataset),
            "val_segments": 0 if val_dataset is None else len(val_dataset),
            "loader_batches_per_process": len(train_loader),
            "batch_size_per_process": batch_size,
            "embedding_dim": _unwrap_module(model, accelerator).embedding_dim,
            "segment_repair": repair_cfg,
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
                    model.train()
                    losses = train_segment_repair_ae_step(
                        model,
                        optimizer,
                        batch["audio"].float().to(device, non_blocking=True),
                        batch["dred_audio"].float().to(device, non_blocking=True),
                        batch["loss_mask"].bool().to(device, non_blocking=True),
                        sample_rate=cfg["sample_rate"],
                        chunk_ms=cfg["chunk_ms"],
                        mse_weight=float(repair_cfg.get("mse_weight", 1.0)),
                        l1_weight=float(repair_cfg.get("l1_weight", 0.5)),
                        stft_weight=float(repair_cfg.get("stft_weight", 0.5)),
                        accelerator=accelerator,
                    )
                    step += 1
                    metrics = {
                        "total": float(losses.total),
                        "mse": float(losses.mse),
                        "l1": float(losses.l1),
                        "stft": float(losses.stft),
                        "lost_frames": float(losses.lost_frames),
                    }
                    if progress is not None:
                        progress.update(1)
                        _progress_set_postfix(progress, {"loss": metrics["total"], "mse": metrics["mse"], "lost": metrics["lost_frames"]})
                    if step % args.log_every == 0:
                        logger.log({"event": "metrics", "split": "train", "stage": "repair", "epoch": epoch, "step": step, **metrics})
                        tb_logger.report("train", metrics, step, epoch)
                    if step % args.save_every == 0 or step >= target_steps:
                        checkpoint_path = out_dir / f"repair_step_{step}.pt"
                        _save_checkpoint(
                            checkpoint_path,
                            {
                                "step": step,
                                "epoch": epoch,
                                "segment_ae_state_dict": _state_dict(model, accelerator),
                                "segment_encoder_state_dict": _state_dict(_unwrap_module(model, accelerator).encoder),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "config": cfg,
                            },
                            accelerator,
                        )
                        logger.log({"event": "checkpoint", "stage": "repair", "epoch": epoch, "step": step, "path": str(checkpoint_path)})
                    if step >= target_steps:
                        break
            finally:
                if progress is not None:
                    progress.close()

            if _should_validate(args, epoch, step >= target_steps):
                val_metrics = validate_repair(model, val_loader, device, cfg, accelerator, args.val_max_batches)
                if val_metrics is not None:
                    logger.log({"event": "validation", "split": "val", "stage": "repair", "epoch": epoch, "step": step, **val_metrics})
                    tb_logger.report("val", val_metrics, step, epoch)
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
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable TensorBoard logging")
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log root; defaults to <output-dir>/tensorboard")
    parser.add_argument("--clearml", dest="clearml", action=argparse.BooleanOptionalAction, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--clearml-project", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--clearml-task-name", default=None, help=argparse.SUPPRESS)
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
