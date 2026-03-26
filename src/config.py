from dataclasses import dataclass, field
import os


@dataclass
class DataConfig:
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    train_files: str = ""
    val_files: str = ""

    def __post_init__(self):
        if not self.train_files:
            self.train_files = os.path.join(self.data_path, "fineweb_train_*.bin")
        if not self.val_files:
            self.val_files = os.path.join(self.data_path, "fineweb_val_*.bin")


@dataclass
class TrainConfig:
    val_batch_size: int = 524_288
    val_loss_every: int = 1000
    train_log_every: int = 200

    iterations: int = 20_000
    warmdown_iters: int = 1200
    warmup_steps: int = 20

    train_batch_tokens: int = 524_288
    train_seq_len: int = 1024
    max_wallclock_seconds: float = 600.0


@dataclass
class OptimizerConfig:
    embed_lr: float = 0.6
    head_lr: float = 0.008
    tied_embed_lr: float = 0.05
    matrix_lr: float = 0.04
    scalar_lr: float = 0.04

    muon_momentum: float = 0.95
    muon_backend_steps: int = 5
    muon_momentum_warmup_start: float = 0.85
    muon_momentum_warmup_steps: int = 500

    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip_norm: float = 0.0


@dataclass
class RunConfig:
    run_id: str = ""
    seed: int = 1337
    output_root: str = "artifacts/runs/artifact_probe"
    device: str = "cuda"
    save_quantized_artifact_summary: bool = True


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "parameter-golf"
    entity: str = ""
    mode: str = "online"
    tags: list[str] = field(default_factory=lambda: ["artifact-probe"])


@dataclass
class ModelConfig:
    vocab_size: int = 1024
    num_layers: int = 9
    model_dim: int = 512

    head_dim: int = 64
    gqa_ratio: int = 2
    auto_derive_attention_shapes: bool = True

    num_heads: int = 8
    num_kv_heads: int = 4

    mlp_mult: int = 2
    tie_embeddings: bool = True
    tied_embed_init_std: float = 0.005
    logit_softcap: float = 30.0
    rope_base: float = 10000.0
    qk_gain_init: float = 1.5
    dtype: str = "bfloat16"

    block_type: str = "baseline"
    attention_type: str = "gqa"
    mlp_type: str = "relu2"
    norm_type: str = "rmsnorm"


@dataclass
class CodeConfig:
    enabled: bool = True
    manifest_name: str = "code_manifest.txt"
    include_config_code: bool = True
    include_optimizer_code: bool = False
    optimizer_file: str = "src/model/optimizer.py"


@dataclass
class QuantConfig:
    enabled: bool = True
    zlib_level: int = 9
    artifact_name: str = "final_model.int8.ptz"
    write_quant_artifact_to_disk: bool = True

    keep_float_fp32_name_patterns: list[str] = field(
        default_factory=lambda: [
            "attn_scale",
            "attn_scales",
            "mlp_scale",
            "mlp_scales",
            "resid_mix",
            "resid_mixes",
            "q_gain",
            "skip_weight",
            "skip_weights",
        ]
    )
    keep_float_max_numel: int = 65_536
    keep_float_store_dtype: str = "float16"
    per_row_scale_dtype: str = "float16"
    clip_percentile: float = 99.99984

@dataclass
class ArtifactProbeConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class TrainExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)