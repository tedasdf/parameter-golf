from __future__ import annotations

import argparse
import inspect
import io
import json
import os
import random
import time
import uuid
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import Tensor, nn

from helper.quantiser import dequantize_state_dict_int8, quantize_state_dict_int8
from config import ArtifactProbeConfig, ModelConfig, QuantConfig
from model_builder import build_model
try:
    from constant import MANIFEST_TEXT
    from model.gpt import GPT
    from model.registry import (
        get_attention_cls,
        get_block_cls,
        get_mlp_cls,
        get_norm_cls,
    )
except ImportError:
    from constant import MANIFEST_TEXT
    from model.gpt import GPT
    from model.registry import (
        get_attention_cls,
        get_block_cls,
        get_mlp_cls,
        get_norm_cls,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# Generic helpers
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Artifact probe for model + quantization.")
    parser.add_argument("--config_path", type=str, required=False)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.num_layers=12 quant.clip_percentile=99.99",
    )
    return parser.parse_args()


def load_cfg(config_path: str | None, overrides: list[str]) -> ArtifactProbeConfig:
    base = OmegaConf.structured(ArtifactProbeConfig())

    if config_path:
        user_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(base, user_cfg)
    else:
        cfg = base

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    cfg_obj = OmegaConf.to_object(cfg)
    if not isinstance(cfg_obj, ArtifactProbeConfig):
        raise TypeError("Merged config did not resolve to ArtifactProbeConfig")

    if not cfg_obj.wandb.enabled:
        cfg_obj.wandb.mode = "disabled"

    return cfg_obj


def to_flat_dict(cfg: ArtifactProbeConfig) -> dict[str, Any]:
    raw = asdict(cfg)
    flat: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                next_prefix = f"{prefix}.{k}" if prefix else k
                _walk(next_prefix, v)
        else:
            flat[prefix] = value

    _walk("", raw)
    return flat


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_master_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device(device_str)


# ============================================================
# Manifest helpers
# ============================================================

def _to_repo_relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _get_source_path(obj: Any) -> Path | None:
    src = inspect.getsourcefile(obj) or inspect.getfile(obj)
    if src is None:
        return None
    return Path(src).resolve()


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def read_text_if_exists(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def get_model_source_files_from_config(model_cfg: ModelConfig) -> list[str]:
    block_cls = get_block_cls(model_cfg.block_type)
    attention_cls = get_attention_cls(model_cfg.attention_type)
    mlp_cls = get_mlp_cls(model_cfg.mlp_type)
    norm_cls = get_norm_cls(model_cfg.norm_type)

    candidate_paths: list[Path | None] = [
        _get_source_path(GPT),
        _get_source_path(block_cls),
        _get_source_path(attention_cls),
        _get_source_path(mlp_cls),
        _get_source_path(norm_cls),
        _get_source_path(get_block_cls),  # register.py
    ]

    rel_files = [
        _to_repo_relative(path)
        for path in candidate_paths
        if path is not None
    ]
    return _dedupe_keep_order(rel_files)


def build_model_code_blob_from_config(model_cfg: ModelConfig) -> str:
    files = get_model_source_files_from_config(model_cfg)

    chunks: list[str] = []
    for file_str in files:
        path = REPO_ROOT / file_str
        if not path.exists():
            chunks.append(f"===== {file_str} =====\n[MISSING FILE]\n")
            continue
        chunks.append(f"===== {file_str} =====\n{path.read_text(encoding='utf-8')}\n")

    return "\n".join(chunks)


def render_code_manifest(cfg: ArtifactProbeConfig) -> str:
    config_code = ""
    if cfg.code.include_config_code:
        config_code = read_text_if_exists(str(REPO_ROOT / "src/config.py"))

    optimizer_code = ""
    if cfg.code.include_optimizer_code:
        optimizer_code = read_text_if_exists(str(REPO_ROOT / cfg.code.optimizer_file))

    model_code = build_model_code_blob_from_config(cfg.model)

    rendered = (
        MANIFEST_TEXT
        .replace("<<CONFIG_CODE>>", config_code)
        .replace("<<OPTIMIZER CODE>>", optimizer_code)
        .replace("<<MODEL_CODE>>", model_code)
    )
    return rendered


def write_rendered_manifest(cfg: ArtifactProbeConfig, out_path: Path) -> int:
    rendered = render_code_manifest(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return len(rendered.encode("utf-8"))


# ============================================================
# Quantization
# ============================================================


def compare_state_dicts(
    original: dict[str, Tensor],
    restored: dict[str, Tensor],
    eps: float = 1e-12,
) -> dict[str, float]:
    max_abs = 0.0
    sum_abs = 0.0
    total_numel = 0

    diff_sq_sum = 0.0
    ref_sq_sum = 0.0
    max_rel = 0.0

    for name, orig in original.items():
        orig_cpu = orig.detach().to("cpu").contiguous()
        if not orig_cpu.is_floating_point():
            continue

        rec = restored[name].detach().to("cpu").contiguous()

        ref = orig_cpu.float()
        got = rec.float()
        diff = (ref - got).abs()

        if diff.numel() > 0:
            max_abs = max(max_abs, float(diff.max().item()))
            max_rel = max(
                max_rel,
                float((diff / ref.abs().clamp_min(eps)).max().item()),
            )

        sum_abs += float(diff.sum().item())
        total_numel += diff.numel()

        diff_sq_sum += float((ref - got).pow(2).sum().item())
        ref_sq_sum += float(ref.pow(2).sum().item())

    mean_abs = sum_abs / max(total_numel, 1)
    rel_l2 = (diff_sq_sum**0.5) / max(ref_sq_sum**0.5, eps)

    return {
        "quant/error_max_abs": max_abs,
        "quant/error_mean_abs": mean_abs,
        "quant/error_rel_l2": rel_l2,
        "quant/error_max_rel": max_rel,
    }


def probe_quantized_artifact(
    model: nn.Module,
    run_dir: Path,
    code_bytes: int,
    quant_cfg: QuantConfig,
    master_process: bool,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "quant/int8_payload_bytes": 0,
        "quant/quant_raw_bytes": 0,
        "quant/compressed_artifact_bytes": 0,
        "quant/compressed_submission_bytes": 0,
        "quant/payload_ratio": 0.0,
        "quant/error_max_abs": 0.0,
        "quant/error_mean_abs": 0.0,
        "quant/error_rel_l2": 0.0,
        "quant/error_max_rel": 0.0,
    }

    if not quant_cfg.enabled:
        return metrics

    original_state = {
        k: v.detach().to("cpu").contiguous()
        for k, v in model.state_dict().items()
    }
    quant_obj, quant_stats = quantize_state_dict_int8(original_state, quant_cfg)
    restored_state = dequantize_state_dict_int8(quant_obj)
    error_metrics = compare_state_dicts(original_state, restored_state)

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=quant_cfg.zlib_level)

    compressed_artifact_bytes = len(quant_blob)
    if quant_cfg.write_quant_artifact_to_disk and master_process:
        out_path = run_dir / quant_cfg.artifact_name
        out_path.write_bytes(quant_blob)
        compressed_artifact_bytes = out_path.stat().st_size

    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    compressed_submission_bytes = compressed_artifact_bytes + code_bytes

    metrics.update(
        {
            "quant/int8_payload_bytes": int(quant_stats["int8_payload_bytes"]),
            "quant/quant_raw_bytes": int(len(quant_raw)),
            "quant/compressed_artifact_bytes": int(compressed_artifact_bytes),
            "quant/compressed_submission_bytes": int(compressed_submission_bytes),
            "quant/payload_ratio": float(ratio),
            **error_metrics,
        }
    )
    return metrics


# ============================================================
# Main
# ============================================================

def main() -> None:
    cli_args = parse_args()
    cfg = load_cfg(cli_args.config_path, cli_args.overrides)

    seed_everything(cfg.run.seed)
    master_process = is_master_process()

    wandb_run = None
    if master_process:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            config=asdict(cfg),
        )

    if not cfg.run.run_id:
        cfg.run.run_id = (
            wandb_run.id if wandb_run is not None else f"artifact-probe-{uuid.uuid4().hex[:8]}"
        )

    run_dir = REPO_ROOT / cfg.run.output_root / cfg.run.run_id
    if master_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.structured(cfg)),
            encoding="utf-8",
        )
        if wandb_run is not None:
            wandb.config.update({"run.run_id": cfg.run.run_id}, allow_val_change=True)

    device = resolve_device(cfg.run.device)

    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg.model, device)
    model.eval()

    init_wallclock_sec = time.perf_counter() - t0

    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    manifest_path = run_dir / cfg.code.manifest_name
    code_manifest_bytes = 0
    if cfg.code.enabled:
        if master_process:
            code_manifest_bytes = write_rendered_manifest(cfg, manifest_path)
        else:
            rendered = render_code_manifest(cfg)
            code_manifest_bytes = len(rendered.encode("utf-8"))

    quant_metrics = probe_quantized_artifact(
        model=model,
        run_dir=run_dir,
        code_bytes=code_manifest_bytes,
        quant_cfg=cfg.quant,
        master_process=master_process,
    )

    summary = {
        "run_id": cfg.run.run_id,
        "model/param_count": int(param_count),
        "model/trainable_param_count": int(trainable_param_count),
        **quant_metrics,
    }

    if master_process:
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if wandb_run is not None:
            wandb.log(summary)
            for k, v in summary.items():
                if isinstance(v, (int, float, bool, str)):
                    wandb_run.summary[k] = v
            wandb_run.finish()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()