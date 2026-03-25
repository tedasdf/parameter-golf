from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from constant import MANIFEST_TEXT
import numpy as np
from quantiser import quantize_state_dict_int8
import torch
import wandb
from omegaconf import OmegaConf
from torch import Tensor

# ------------------------------------------------------------
# Imports
# Adjust automatically depending on whether you run from repo root
# or from inside src-style imports.
# ------------------------------------------------------------
try:
    from model.gpt import GPT, CastedLinear, restore_low_dim_params_to_fp32
    from constant import (
        INT8_CLIP_Q,
        INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
        INT8_KEEP_FLOAT_MAX_NUMEL,
        INT8_KEEP_FLOAT_STORE_DTYPE,
        INT8_PER_ROW_SCALE_DTYPE,
    )
except ImportError:
    from model.gpt import GPT, CastedLinear, restore_low_dim_params_to_fp32
    from constant import (
        INT8_CLIP_Q,
        INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
        INT8_KEEP_FLOAT_MAX_NUMEL,
        INT8_KEEP_FLOAT_STORE_DTYPE,
        INT8_PER_ROW_SCALE_DTYPE,
    )


# ============================================================
# Structured config
# ============================================================

@dataclass
class RunConfig:
    run_id: str = ""
    seed: int = 1337
    output_root: str = "artifacts/runs/artifact_probe"
    device: str = "cuda"
    save_raw_state_dict: bool = False


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "parameter-golf"
    entity: str = ""
    mode: str = "online"  # online / offline / disabled
    tags: list[str] = field(default_factory=lambda: ["artifact-probe"])


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    num_layers: int = 9
    model_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 4
    mlp_mult: float = 2.0
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.5
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.0
    dtype: str = "bfloat16"  # float32 / float16 / bfloat16


@dataclass
class CodeConfig:
    enabled: bool = True
    manifest_name: str = "code_manifest.txt"

    config_file: str = "src/config.py"
    model_file: str = "src/model/gpt.py"
    optimizer_file: str = "src/model/optimizer.py"

    include_optimizer_code: bool = False


@dataclass
class QuantConfig:
    enabled: bool = True
    zlib_level: int = 9
    artifact_name: str = "final_model.int8.ptz"
    write_quant_artifact_to_disk: bool = True


@dataclass
class ArtifactProbeConfig:
    run: RunConfig = field(default_factory=RunConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)


# ============================================================
# CLI / config helpers
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe artifact size for a GPT config.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides. Example: model.num_layers=12 model.model_dim=640",
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

    if not cfg_obj.run.run_id:
        cfg_obj.run.run_id = f"artifact-probe-{uuid.uuid4().hex[:8]}"

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


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device(device_str)


def is_master_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


# ============================================================
# Model helpers
# ============================================================

def apply_model_dtype(model: torch.nn.Module, dtype_name: str) -> torch.nn.Module:
    dtype_name = dtype_name.lower()
    if dtype_name == "float32":
        return model.float()
    if dtype_name == "float16":
        return model.half()
    if dtype_name == "bfloat16":
        return model.bfloat16()
    raise ValueError(f"Unsupported dtype: {dtype_name}")
    return 

def build_model(cfg: ArtifactProbeConfig, device: torch.device) -> torch.nn.Module:
    model = GPT(
        vocab_size=cfg.model.vocab_size,
        num_layers=cfg.model.num_layers,
        model_dim=cfg.model.model_dim,
        num_heads=cfg.model.num_heads,
        num_kv_heads=cfg.model.num_kv_heads,
        mlp_mult=cfg.model.mlp_mult,
        tie_embeddings=cfg.model.tie_embeddings,
        tied_embed_init_std=cfg.model.tied_embed_init_std,
        logit_softcap=cfg.model.logit_softcap,
        rope_base=cfg.model.rope_base,
        qk_gain_init=cfg.model.qk_gain_init,
    )

    model = model.to(device)
    model = apply_model_dtype(model, cfg.model.dtype)

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    return model


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def get_logical_bytes(model: torch.nn.Module) -> tuple[int, int]:
    param_bytes = sum(tensor_nbytes(p) for p in model.parameters())
    buffer_bytes = sum(tensor_nbytes(b) for b in model.buffers())
    return int(param_bytes), int(buffer_bytes)


def get_serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    state_dict_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state_dict_cpu, buf)
    return len(buf.getvalue())


# ============================================================
# Code manifest helpers
# ============================================================

def read_text_if_exists(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def render_code_manifest(cfg: CodeConfig) -> str:
    config_code = read_text_if_exists(cfg.config_file)
    model_code = read_text_if_exists(cfg.model_file)
    optimizer_code = read_text_if_exists(cfg.optimizer_file) if cfg.include_optimizer_code else ""

    rendered = (
        MANIFEST_TEXT
        .replace("<<CONFIG_CODE>>", config_code)
        .replace("<<OPTIMIZER_CODE>>", optimizer_code)
        .replace("<<MODEL_CODE>>", model_code)
    )
    return rendered


def write_rendered_manifest(cfg: CodeConfig, out_path: Path) -> int:
    rendered = render_code_manifest(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return len(rendered.encode("utf-8"))

# ============================================================
# Quantization helpers
# Reused from your current export logic
# ============================================================

def probe_quantized_artifact(
    model: torch.nn.Module,
    run_dir: Path,
    code_bytes: int,
    cfg: QuantConfig,
) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {
        "artifact/int8_payload_bytes": 0,
        "artifact/quant_raw_bytes": 0,
        "artifact/compressed_artifact_bytes": 0,
        "artifact/compressed_submission_bytes": 0,
        "artifact/payload_ratio": 0.0,
    }

    if not cfg.enabled:
        return metrics

    quant_obj, quant_stats = quantize_state_dict_int8(model.state_dict())

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()

    quant_blob = zlib.compress(quant_raw, level=cfg.zlib_level)
    quant_raw_bytes = len(quant_raw)
    compressed_artifact_bytes = len(quant_blob)

    if cfg.write_quant_artifact_to_disk:
        out_path = run_dir / cfg.artifact_name
        out_path.write_bytes(quant_blob)
        compressed_artifact_bytes = out_path.stat().st_size

    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    compressed_submission_bytes = compressed_artifact_bytes + code_bytes

    metrics.update(
        {
            "artifact/int8_payload_bytes": int(quant_stats["int8_payload_bytes"]),
            "artifact/quant_raw_bytes": int(quant_raw_bytes),
            "artifact/compressed_artifact_bytes": int(compressed_artifact_bytes),
            "artifact/compressed_submission_bytes": int(compressed_submission_bytes),
            "artifact/payload_ratio": float(ratio),
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

    run_dir = Path(cfg.run.output_root) / cfg.run.run_id
    if master_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(OmegaConf.structured(cfg)),
            encoding="utf-8",
        )

    device = resolve_device(cfg.run.device)

    wandb_run = None
    if master_process:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=cfg.run.run_id,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
            config=to_flat_dict(cfg),
        )

    t0 = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(cfg, device)
    model.eval()

    init_wallclock_sec = time.perf_counter() - t0

    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_count = sum(b.numel() for b in model.buffers())

    param_bytes, buffer_bytes = get_logical_bytes(model)
    raw_model_bytes = get_serialized_state_dict_bytes(model)
    manifest_path = run_dir / cfg.code.manifest_name
    code_manifest_bytes = 0

    if cfg.code.enabled:
        if master_process:
            code_manifest_bytes = write_rendered_manifest(cfg.code, manifest_path)
        else:
            rendered = render_code_manifest(cfg.code)
            code_manifest_bytes = len(rendered.encode("utf-8"))


    raw_submission_bytes = raw_model_bytes + code_manifest_bytes

    quant_metrics = probe_quantized_artifact(
        model=model,
        run_dir=run_dir,
        code_bytes=code_manifest_bytes,
        cfg=cfg.quant,
    )

    cuda_allocated_mb = 0.0
    cuda_reserved_mb = 0.0
    cuda_peak_allocated_mb = 0.0
    cuda_peak_reserved_mb = 0.0

    if device.type == "cuda":
        cuda_allocated_mb = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
        cuda_reserved_mb = float(torch.cuda.memory_reserved(device) / (1024 ** 2))
        cuda_peak_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        cuda_peak_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

    if master_process and cfg.run.save_raw_state_dict:
        raw_state_path = run_dir / "final_model.pt"
        state_dict_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        torch.save(state_dict_cpu, raw_state_path)

    summary = {
        "run_id": cfg.run.run_id,
        "device": str(device),
        "model/param_count": int(param_count),
        "model/trainable_param_count": int(trainable_param_count),
        "model/buffer_count": int(buffer_count),
        "model/param_bytes_logical": int(param_bytes),
        "model/buffer_bytes_logical": int(buffer_bytes),
        "artifact/raw_model_bytes": int(raw_model_bytes),
        "artifact/code_manifest_bytes": int(code_manifest_bytes),
        "artifact/code_manifest_bytes": int(code_manifest_bytes),
        "artifact/raw_submission_bytes": int(raw_submission_bytes),
        "runtime/init_wallclock_sec": float(init_wallclock_sec),
        "memory/cuda_allocated_mb_after_init": float(cuda_allocated_mb),
        "memory/cuda_reserved_mb_after_init": float(cuda_reserved_mb),
        "memory/cuda_peak_allocated_mb_after_init": float(cuda_peak_allocated_mb),
        "memory/cuda_peak_reserved_mb_after_init": float(cuda_peak_reserved_mb),
        "model/tie_embeddings": bool(cfg.model.tie_embeddings),
        "model/num_layers": int(cfg.model.num_layers),
        "model/model_dim": int(cfg.model.model_dim),
        "model/num_heads": int(cfg.model.num_heads),
        "model/num_kv_heads": int(cfg.model.num_kv_heads),
        "model/mlp_mult": float(cfg.model.mlp_mult),
        "model/logit_softcap": float(cfg.model.logit_softcap),
        "model/dtype": cfg.model.dtype,
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