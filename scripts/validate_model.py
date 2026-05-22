from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import soundfile as sf
import torch
import torchaudio.functional as AF
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from streamloss_codec.cache.dred_cache import CachedDredDataset  # noqa: E402
from streamloss_codec.codec import StreamingSpeechCodec, chunk_samples, frame_audio  # noqa: E402
from streamloss_codec.config import load_config  # noqa: E402
from streamloss_codec.state_repair import StateRepairMiniEncoder  # noqa: E402


def _model_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg["model"].items() if key != "active_quantizers_train"}


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return device


def _load_checkpoint(path: str, codec: StreamingSpeechCodec, repair: StateRepairMiniEncoder) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    codec.load_state_dict(checkpoint["codec_state_dict"], strict=False)
    if "repair_state_dict" in checkpoint:
        repair.load_state_dict(checkpoint["repair_state_dict"], strict=False)
    return checkpoint


def _count_parameters(module: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return float(ordered[low])
    weight = index - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _summarize(values: list[float]) -> dict[str, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"mean": None, "median": None, "p05": None, "p95": None}
    return {
        "mean": float(mean(finite)),
        "median": float(median(finite)),
        "p05": _percentile(finite, 0.05),
        "p95": _percentile(finite, 0.95),
    }


def _write_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.detach().cpu().clamp(-1, 1).numpy(), sample_rate)


def _write_48k_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    if sample_rate != 48_000:
        audio = AF.resample(audio.detach().cpu(), sample_rate, 48_000)
    _write_wav(path, audio, 48_000)


def _pesq_score(reference: torch.Tensor, degraded: torch.Tensor, sample_rate: int) -> float | None:
    try:
        from pesq import pesq
    except ImportError:
        return None
    if sample_rate != 16_000:
        reference = AF.resample(reference.detach().cpu(), sample_rate, 16_000)
        degraded = AF.resample(degraded.detach().cpu(), sample_rate, 16_000)
        sample_rate = 16_000
    try:
        return float(pesq(sample_rate, reference.cpu().numpy(), degraded.cpu().numpy(), "wb"))
    except Exception:
        return None


def _parse_score(stdout: str) -> float | None:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("nisqa_s", "nisqa", "mos", "score"):
            if key in payload:
                return float(payload[key])
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", stdout)
    return float(match.group(0)) if match else None


def _nisqa_score(command_template: str | None, wav_48k: Path) -> float | None:
    if command_template is None:
        return None
    command = shlex.split(command_template.format(wav=str(wav_48k)))
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return _parse_score(result.stdout)


class NisqaSScorer:
    def __init__(self, root: Path, yaml_path: Path | None = None, device: str | None = None) -> None:
        self.root = root
        config_path = yaml_path or root / "config" / "nisqa_s.yaml"
        with config_path.open("r", encoding="utf-8") as handle:
            self.args = yaml.load(handle, Loader=yaml.FullLoader)

        checkpoint = Path(self.args["ckp"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        self.args["ckp"] = str(checkpoint)
        if device is not None:
            self.args["inf_device"] = device

        sys.path.insert(0, str(root))
        from src.core.model_torch import model_init
        from src.utils.process_utils import process

        self.process = process
        original_torch_load = torch.load

        def trusted_load(*load_args, **load_kwargs):
            load_kwargs.setdefault("weights_only", False)
            return original_torch_load(*load_args, **load_kwargs)

        try:
            torch.load = trusted_load
            self.model, self.h0, self.c0 = model_init(self.args)
        finally:
            torch.load = original_torch_load

    def score(self, wav_48k: Path) -> float:
        audio, sample_rate = sf.read(wav_48k)
        framesize = int(sample_rate * self.args["frame"])
        audio = torch.as_tensor(audio)
        remainder = audio.shape[0] % framesize
        if remainder:
            audio = torch.cat((audio, torch.zeros(framesize - remainder)))
        chunks = torch.split(audio, framesize, dim=0)

        h0 = self.h0.clone()
        c0 = self.c0.clone()
        outputs = []
        with contextlib.redirect_stdout(io.StringIO()):
            if self.args["warmup"]:
                _, _, _ = self.process(torch.zeros((1, framesize)), sample_rate, self.model, h0, c0, self.args)
            for chunk in chunks:
                out, h0, c0 = self.process(chunk, sample_rate, self.model, h0, c0, self.args)
                outputs.append(out[0].detach().cpu())
        return float(torch.stack(outputs, dim=0).mean(dim=0)[0].item())


def _make_nisqa_scorer(args: argparse.Namespace) -> NisqaSScorer | None:
    if args.nisqa_command is not None:
        return None
    root = Path(args.nisqa_root)
    if not root.is_dir():
        return None
    yaml_path = Path(args.nisqa_yaml) if args.nisqa_yaml is not None else None
    return NisqaSScorer(root=root, yaml_path=yaml_path, device=args.nisqa_device)


def _score_nisqa(scorer: NisqaSScorer | None, command_template: str | None, wav_48k: Path) -> float | None:
    if scorer is not None:
        return scorer.score(wav_48k)
    return _nisqa_score(command_template, wav_48k)


def _raw_packet_bits(packet, codebook_size: int, scalar_quantizers: int) -> int:
    bits = 0
    vector_bits = int(math.ceil(math.log2(codebook_size)))
    for idx, indices in enumerate(packet.indices):
        if idx < scalar_quantizers:
            bits += int(indices.numel()) * 4
        else:
            bits += int(indices.numel()) * vector_bits
    return bits


@torch.inference_mode()
def _decode_imsit_segment(
    codec: StreamingSpeechCodec,
    repair: StateRepairMiniEncoder | None,
    audio: torch.Tensor,
    dred_audio: torch.Tensor,
    loss_mask: torch.Tensor,
    active_quantizers: int,
    sample_rate: int,
    chunk_ms: int,
    codebook_size: int,
    scalar_quantizers: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_size = chunk_samples(sample_rate, chunk_ms)
    frames = frame_audio(audio, frame_size)
    dred_frames = frame_audio(dred_audio, frame_size)
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)
    if loss_mask.shape != frames.shape[:2]:
        raise ValueError(f"loss_mask must have shape {frames.shape[:2]}, got {loss_mask.shape}")

    state = None
    outputs = []
    packet_bits = 0
    encoded_packets = 0
    latent_dim = codec.quantizer.scalar[0].from_scalar.out_channels
    latent_frames = frame_size // 4

    started = time.perf_counter()
    for frame_idx in range(frames.shape[1]):
        chunk = frames[:, frame_idx].to(device)
        packet, _ = codec.encode_chunk(chunk, active_quantizers=active_quantizers)
        packet_bits += _raw_packet_bits(packet, codebook_size, scalar_quantizers)
        encoded_packets += 1
        normal_decoded, normal_state = codec.decode_packet(packet, state)
        if state is None:
            state = codec.decoder.initial_state(chunk.shape[0], chunk.device)

        lost = loss_mask[:, frame_idx].to(device).bool()
        if lost.any():
            if repair is not None:
                repaired_state = repair(dred_frames[:, frame_idx].to(device), state)
            else:
                repaired_state = state
            z_lost = torch.zeros(chunk.shape[0], latent_dim, latent_frames, device=device)
            lost_decoded, lost_state = codec.decoder(z_lost, repaired_state)
            next_h = torch.where(lost.view(1, -1, 1), lost_state.gru_h, normal_state.gru_h)
            state = type(normal_state)(gru_h=next_h)
            decoded = torch.where(lost.view(-1, 1), lost_decoded[:, :frame_size], normal_decoded[:, :frame_size])
        else:
            state = normal_state
            decoded = normal_decoded[:, :frame_size]
        outputs.append(decoded.cpu())
    elapsed = time.perf_counter() - started

    output = torch.cat(outputs, dim=-1)[0, : audio.numel()]
    audio_seconds = audio.numel() / sample_rate
    return output, {
        "elapsed_seconds": float(elapsed),
        "rtf": float(elapsed / max(audio_seconds, 1e-12)),
        "chunks_per_second": float(encoded_packets / max(elapsed, 1e-12)),
        "raw_payload_bits": float(packet_bits),
        "raw_payload_kbps": float(packet_bits / max(audio_seconds, 1e-12) / 1000.0),
    }


def _make_row(
    *,
    segment_index: int,
    system: str,
    pesq_wb: float | None,
    nisqa_s: float | None,
    bitrate_kbps: float | None,
    speed: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "segment_index": segment_index,
        "system": system,
        "pesq_wb": pesq_wb,
        "nisqa_s": nisqa_s,
        "bitrate_kbps": bitrate_kbps,
        "rtf": None if speed is None else speed["rtf"],
        "chunks_per_second": None if speed is None else speed["chunks_per_second"],
        "elapsed_seconds": None if speed is None else speed["elapsed_seconds"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["segment_index", "system", "pesq_wb", "nisqa_s", "bitrate_kbps", "rtf", "chunks_per_second", "elapsed_seconds"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems = sorted({row["system"] for row in rows})
    summary: dict[str, Any] = {}
    for system in systems:
        selected = [row for row in rows if row["system"] == system]
        summary[system] = {
            "segments": len(selected),
            "pesq_wb": _summarize([row["pesq_wb"] for row in selected if row["pesq_wb"] is not None]),
            "nisqa_s": _summarize([row["nisqa_s"] for row in selected if row["nisqa_s"] is not None]),
            "bitrate_kbps": _summarize([row["bitrate_kbps"] for row in selected if row["bitrate_kbps"] is not None]),
            "rtf": _summarize([row["rtf"] for row in selected if row["rtf"] is not None]),
            "chunks_per_second": _summarize([row["chunks_per_second"] for row in selected if row["chunks_per_second"] is not None]),
        }
    return summary


def _write_summary_md(path: Path, summary: dict[str, Any], params: dict[str, Any], metadata: dict[str, Any]) -> None:
    lines = [
        "# Validation Summary",
        "",
        "## Metadata",
        "",
        f"- checkpoint: `{metadata['checkpoint']}`",
        f"- manifest: `{metadata['manifest']}`",
        f"- active_quantizers: `{metadata['active_quantizers']}`",
        f"- sample_rate: `{metadata['sample_rate']}`",
        f"- chunk_ms: `{metadata['chunk_ms']}`",
        "",
        "## Parameters",
        "",
        f"- codec total/trainable: `{params['codec']['total']}` / `{params['codec']['trainable']}`",
        f"- repair total/trainable: `{params['repair']['total']}` / `{params['repair']['trainable']}`",
        f"- combined total/trainable: `{params['combined']['total']}` / `{params['combined']['trainable']}`",
        "",
        "## Metrics",
        "",
        "| system | segments | PESQ-WB mean | NISQA-S mean | bitrate kbps mean | RTF mean | chunks/s mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system, values in summary.items():
        lines.append(
            "| {system} | {segments} | {pesq} | {nisqa} | {bitrate} | {rtf} | {cps} |".format(
                system=system,
                segments=values["segments"],
                pesq=_format_metric(values["pesq_wb"]["mean"]),
                nisqa=_format_metric(values["nisqa_s"]["mean"]),
                bitrate=_format_metric(values["bitrate_kbps"]["mean"]),
                rtf=_format_metric(values["rtf"]["mean"]),
                cps=_format_metric(values["chunks_per_second"]["mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _progress(iterable, total: int, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc="Validating", unit="segment", dynamic_ncols=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate IMSIT streaming codec against cached Opus DRED outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--active-quantizers", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--output-dir", default="runs/validation/latest")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nisqa-command", default=None, help="External command template for 48 kHz wavs; use {wav} as the file placeholder.")
    parser.add_argument("--nisqa-root", default="NISQA-s", help="Local NISQA-S checkout used when --nisqa-command is not set.")
    parser.add_argument("--nisqa-yaml", default=None, help="Override local NISQA-S yaml config.")
    parser.add_argument("--nisqa-device", default="cpu", help="NISQA-S inference device for local scorer.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-audio", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sample_rate = int(cfg["sample_rate"])
    chunk_ms = int(cfg["chunk_ms"])
    model_cfg = cfg["model"]
    active_quantizers = args.active_quantizers or model_cfg.get("active_quantizers_train", [model_cfg["scalar_quantizers"] + model_cfg["vector_quantizers"]])[-1]
    manifest = args.manifest or cfg["dred_cache"]["val_manifest_path"]
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    wav48_dir = output_dir / "nisqa_48k"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    codec = StreamingSpeechCodec(**_model_kwargs(cfg))
    repair = StateRepairMiniEncoder(decoder_hidden=model_cfg["decoder_hidden"])
    checkpoint = _load_checkpoint(args.checkpoint, codec, repair)
    has_repair = "repair_state_dict" in checkpoint
    codec.to(device).eval()
    repair.to(device).eval()
    repair_for_decode = repair if has_repair else None

    params = {
        "codec": _count_parameters(codec),
        "repair": _count_parameters(repair),
    }
    params["combined"] = {
        "total": params["codec"]["total"] + params["repair"]["total"],
        "trainable": params["codec"]["trainable"] + params["repair"]["trainable"],
    }

    dataset = CachedDredDataset(manifest)
    limit = len(dataset) if args.max_segments is None else min(args.max_segments, len(dataset))
    nisqa_scorer = _make_nisqa_scorer(args)
    rows: list[dict[str, Any]] = []
    for index in _progress(range(limit), total=limit, enabled=args.progress):
        item = dataset[index]
        reference = item.audio.float()
        opus_dred = item.dred_audio.float()[: reference.numel()]
        loss_mask = item.loss_mask.bool()
        imsit_output, imsit_speed = _decode_imsit_segment(
            codec,
            repair_for_decode,
            reference.unsqueeze(0),
            opus_dred.unsqueeze(0),
            loss_mask,
            active_quantizers,
            sample_rate,
            chunk_ms,
            int(model_cfg["codebook_size"]),
            int(model_cfg["scalar_quantizers"]),
            device,
        )

        segment_dir = audio_dir / f"segment_{item.segment_index:09d}"
        wav48_segment_dir = wav48_dir / f"segment_{item.segment_index:09d}"
        if args.save_audio:
            _write_wav(segment_dir / "input.wav", reference, sample_rate)
            _write_wav(segment_dir / "imsit.wav", imsit_output, sample_rate)
            _write_wav(segment_dir / "opus_dred.wav", opus_dred, sample_rate)
            _write_48k_wav(wav48_segment_dir / "imsit.wav", imsit_output, sample_rate)
            _write_48k_wav(wav48_segment_dir / "opus_dred.wav", opus_dred, sample_rate)

        imsit_48k = wav48_segment_dir / "imsit.wav"
        opus_48k = wav48_segment_dir / "opus_dred.wav"
        if not args.save_audio:
            _write_48k_wav(imsit_48k, imsit_output, sample_rate)
            _write_48k_wav(opus_48k, opus_dred, sample_rate)

        rows.append(
            _make_row(
                segment_index=item.segment_index,
                system="imsit",
                pesq_wb=_pesq_score(reference, imsit_output, sample_rate),
                nisqa_s=_score_nisqa(nisqa_scorer, args.nisqa_command, imsit_48k),
                bitrate_kbps=imsit_speed["raw_payload_kbps"],
                speed=imsit_speed,
            )
        )
        rows.append(
            _make_row(
                segment_index=item.segment_index,
                system="opus_dred",
                pesq_wb=_pesq_score(reference, opus_dred, sample_rate),
                nisqa_s=_score_nisqa(nisqa_scorer, args.nisqa_command, opus_48k),
                bitrate_kbps=float(cfg["dred_cache"].get("bitrate", 64_000)) / 1000.0,
                speed=None,
            )
        )

    summary = _summarize_rows(rows)
    metadata = {
        "checkpoint": args.checkpoint,
        "checkpoint_step": checkpoint.get("step"),
        "has_repair_state": has_repair,
        "manifest": str(manifest),
        "segments": limit,
        "active_quantizers": active_quantizers,
        "sample_rate": sample_rate,
        "chunk_ms": chunk_ms,
        "nisqa_command": args.nisqa_command,
        "nisqa_root": None if nisqa_scorer is None else str(Path(args.nisqa_root)),
        "nisqa_yaml": args.nisqa_yaml,
    }
    payload = {"metadata": metadata, "parameters": params, "summary": summary, "rows": rows}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "metrics.csv", rows)
    _write_summary_md(output_dir / "summary.md", summary, params, metadata)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary, "parameters": params}, sort_keys=True))


if __name__ == "__main__":
    main()
