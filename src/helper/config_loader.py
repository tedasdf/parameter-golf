from __future__ import annotations

import argparse
from pathlib import Path
from typing import Type, TypeVar

from omegaconf import OmegaConf, DictConfig

from config import TrainExperimentConfig, ArtifactProbeConfig


T = TypeVar("T")


def build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    return parser


def _merge_with_schema(schema_cls: Type[T], config_path: str | None, overrides: list[str]) -> T:
    # Start from the dataclass schema defaults
    base_cfg: DictConfig = OmegaConf.structured(schema_cls)

    # Merge YAML if provided
    if config_path is not None:
        yaml_cfg = OmegaConf.load(config_path)
        base_cfg = OmegaConf.merge(base_cfg, yaml_cfg)

    # Merge CLI dotlist overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    # Convert back into nested dataclass object
    obj = OmegaConf.to_object(base_cfg)

    if not isinstance(obj, schema_cls):
        raise TypeError(f"Expected {schema_cls.__name__}, got {type(obj).__name__}")

    return obj


def parse_train_config(argv: list[str] | None = None) -> TrainExperimentConfig:
    parser = build_base_parser()
    args, unknown = parser.parse_known_args(argv)

    cfg = _merge_with_schema(TrainExperimentConfig, args.config, unknown)
    return cfg


def parse_artifact_probe_config(argv: list[str] | None = None) -> ArtifactProbeConfig:
    parser = build_base_parser()
    args, unknown = parser.parse_known_args(argv)

    cfg = _merge_with_schema(ArtifactProbeConfig, args.config, unknown)
    return cfg