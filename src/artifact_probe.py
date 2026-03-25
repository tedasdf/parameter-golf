from __future__ import annotations

import argparse
import io
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from model.gpt import GPT, CastedLinear, restore_low_dim_params_to_fp32


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
    mode: str = "online"   # online / offline / disabled
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
    dtype: str = "bfloat16"   # float32 / float16 / bfloat16


@dataclass
class CodeConfig:
    enabled: bool = True
    manifest_name: str = "code_manifest.txt"

    config_file: str = "src/config.py"
    model_file: str = "src/model/gpt.py"
    optimizer_file: str = "src/model/optimizer.py"

    include_config_code: bool = True
    include_model_code: bool = True
    include_optimizer_code: bool = False


@dataclass
class ArtifactProbeConfig:
    run: RunConfig = field(default_factory=RunConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    code: CodeConfig = field(default_factory=CodeConfig)


# ============================================================
# Helpers
# ============================================================

def get_code_files(cfg: ArtifactProbeConfig) -> list[str]:
    files: list[str] = []

    if not cfg.code.enabled:
        return files

    if cfg.code.include_config_code:
        files.append(cfg.code.config_file)

    if cfg.code.include_model_code:
        files.append(cfg.code.model_file)

    if cfg.code.include_optimizer_code:
        files.append(cfg.code.optimizer_file)

    return files

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe model artifact/memory footprint.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.num_layers=12 model.model_dim=640",
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


def apply_model_dtype(model: torch.nn.Module, dtype_name: str) -> torch.nn.Module:
    dtype_name = dtype_name.lower()

    if dtype_name == "float32":
        model = model.float()
    elif dtype_name == "float16":
        model = model.half()
    elif dtype_name == "bfloat16":
        model = model.bfloat16()
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    return model


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

    # Keep the same pattern you already use in training.
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    return model


def tensor_bytes(t: torch.Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def dtype_breakdown_bytes(model: torch.nn.Module) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype).removeprefix("torch.")
        breakdown[key] = breakdown.get(key, 0) + tensor_bytes(p)
    return breakdown


def buffer_breakdown_bytes(model: torch.nn.Module) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for b in model.buffers():
        key = str(b.dtype).removeprefix("torch.")
        breakdown[key] = breakdown.get(key, 0) + tensor_bytes(b)
    return breakdown


def get_logical_bytes(model: torch.nn.Module) -> tuple[int, int]:
    param_bytes = sum(tensor_bytes(p) for p in model.parameters())
    buffer_bytes = sum(tensor_bytes(b) for b in model.buffers())
    return int(param_bytes), int(buffer_bytes)


def get_serialized_state_dict_bytes(model: torch.nn.Module) -> int:
    state_dict_cpu = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state_dict_cpu, buf)
    return len(buf.getvalue())


def write_code_manifest(files: list[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []
    for file_str in files:
        path = Path(file_str)
        if not path.exists():
            chunks.append(f"===== {file_str} =====\n[MISSING FILE]\n")
            continue

        text = path.read_text(encoding="utf-8")
        chunks.append(f"===== {file_str} =====\n{text}\n")

    manifest = "\n".join(chunks)
    out_path.write_text(manifest, encoding="utf-8")
    return len(manifest.encode("utf-8"))


def count_existing_code_files_bytes(files: list[str]) -> int:
    total = 0
    for file_str in files:
        path = Path(file_str)
        if path.exists():
            total += path.stat().st_size
    return total


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


# ============================================================
# Main
# ============================================================

def main() -> None:
    cli_args = parse_args()
    cfg = load_cfg(cli_args.config_path, cli_args.overrides)

    seed_everything(cfg.run.seed)

    run_dir = Path(cfg.run.output_root) / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    merged_cfg_path = run_dir / "resolved_config.yaml"
    merged_cfg_path.write_text(OmegaConf.to_yaml(OmegaConf.structured(cfg)), encoding="utf-8")

    device = resolve_device(cfg.run.device)

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.run.run_id,
        mode=cfg.wandb.mode,
        tags=cfg.wandb.tags,
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

    param_dtype_bytes = dtype_breakdown_bytes(model)
    buffer_dtype_bytes = buffer_breakdown_bytes(model)

    code_files = get_code_files(cfg)

    code_manifest_bytes = 0
    code_disk_bytes = 0
    manifest_path = run_dir / cfg.code.manifest_name

    if cfg.code.enabled and code_files:
        code_manifest_bytes = write_code_manifest(code_files, manifest_path)
        code_disk_bytes = count_existing_code_files_bytes(code_files)

    raw_submission_bytes = raw_model_bytes + code_manifest_bytes

    cuda_allocated_mb = 0.0
    cuda_reserved_mb = 0.0
    cuda_peak_allocated_mb = 0.0
    cuda_peak_reserved_mb = 0.0

    if device.type == "cuda":
        cuda_allocated_mb = float(torch.cuda.memory_allocated(device) / (1024 ** 2))
        cuda_reserved_mb = float(torch.cuda.memory_reserved(device) / (1024 ** 2))
        cuda_peak_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        cuda_peak_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))

    if cfg.run.save_raw_state_dict:
        raw_state_path = run_dir / "model_state_dict.pt"
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
        "artifact/code_disk_bytes": int(code_disk_bytes),
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
        "model/dtype": cfg.model.dtype,
        "breakdown/param_dtype_bytes": param_dtype_bytes,
        "breakdown/buffer_dtype_bytes": buffer_dtype_bytes,
    }

    for k, v in summary.items():
        if isinstance(v, (int, float, bool, str)):
            wandb_run.summary[k] = v

    wandb.log(
        {
            "model/param_count": int(param_count),
            "model/trainable_param_count": int(trainable_param_count),
            "model/buffer_count": int(buffer_count),
            "model/param_bytes_logical": int(param_bytes),
            "model/buffer_bytes_logical": int(buffer_bytes),
            "artifact/raw_model_bytes": int(raw_model_bytes),
            "artifact/code_manifest_bytes": int(code_manifest_bytes),
            "artifact/code_disk_bytes": int(code_disk_bytes),
            "artifact/raw_submission_bytes": int(raw_submission_bytes),
            "runtime/init_wallclock_sec": float(init_wallclock_sec),
            "memory/cuda_allocated_mb_after_init": float(cuda_allocated_mb),
            "memory/cuda_reserved_mb_after_init": float(cuda_reserved_mb),
            "memory/cuda_peak_allocated_mb_after_init": float(cuda_peak_allocated_mb),
            "memory/cuda_peak_reserved_mb_after_init": float(cuda_peak_reserved_mb),
        }
    )

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    wandb_run.finish()


if __name__ == "__main__":
    main()