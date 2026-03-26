"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
from config import TrainExperimentConfig
from model.gpt import GPT, CastedLinear, restore_low_dim_params_to_fp32
from model.optimiser import Muon
from constant import CONTROL_TENSOR_NAME_PATTERNS, INT8_CLIP_Q, INT8_KEEP_FLOAT_FP32_NAME_PATTERNS, INT8_KEEP_FLOAT_MAX_NUMEL, INT8_KEEP_FLOAT_STORE_DTYPE, INT8_PER_ROW_SCALE_DTYPE
from helper.data_load import DistributedTokenLoader, load_validation_tokens
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import wandb

from helper.quantiser import dequantize_state_dict_int8, quantize_state_dict_int8

# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def eval_val(
    cfg: TrainExperimentConfig,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    train_cfg = cfg.train

    local_batch_tokens = train_cfg.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < train_cfg.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={train_cfg.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={train_cfg.train_seq_len}"
        )

    local_batch_seqs = local_batch_tokens // train_cfg.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // train_cfg.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * train_cfg.train_seq_len
            raw_end = batch_seq_end * train_cfg.train_seq_len + 1

            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, train_cfg.train_seq_len)
            y = local[1:].reshape(-1, train_cfg.train_seq_len)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()

            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()

    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# TRAINING
# -----------------------------
def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")

    cfg = TrainExperimentConfig()
    run_cfg = cfg.run
    data_cfg = cfg.data
    train_cfg = cfg.train
    optim_cfg = cfg.optimizer
    wandb_cfg = cfg.wandb
    model_cfg = cfg.model
    code_cfg = cfg.code
    quant_cfg = cfg.quant

    if not run_cfg.run_id:
        run_cfg.run_id = str(uuid.uuid4())

    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    wandb_run = None
    if master_process:
        wandb_run = wandb.init(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity if wandb_cfg.entity else None,
            name=run_cfg.run_id,
            mode=wandb_cfg.mode if wandb_cfg.enabled else "disabled",
            tags=wandb_cfg.tags,
        )
    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_cfg.run_id}.txt"
        print(logfile)
    
    def wandb_log(payload: dict, step_value: int | None = None) -> None:
        if master_process and wandb_run is not None:
            if step_value is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=step_value)

    print("=" * 100)
    print(
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout
    )
    print("=" * 100)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)
    torch.manual_seed(run_cfg.seed)
    torch.cuda.manual_seed_all(run_cfg.seed)

    if not data_cfg.tokenizer_path.endswith(".model"):
        raise ValueError(
            f"Script only setup for SentencePiece .model file: {data_cfg.tokenizer_path}"
        )

    sp = spm.SentencePieceProcessor(model_file=data_cfg.tokenizer_path)
    if int(sp.vocab_size()) != model_cfg.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={model_cfg.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    dataset_dir = Path(data_cfg.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(data_cfg.val_files, train_cfg.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, model_cfg.vocab_size, device
    )
    
    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=model_cfg.vocab_size,
        num_layers=model_cfg.num_layers,
        model_dim=model_cfg.model_dim,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        mlp_mult=model_cfg.mlp_mult,
        tie_embeddings=model_cfg.tie_embeddings,
        tied_embed_init_std=model_cfg.tied_embed_init_std,
        logit_softcap=model_cfg.logit_softcap,
        rope_base=model_cfg.rope_base,
        qk_gain_init=model_cfg.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    token_lr = optim_cfg.tied_embed_lr if model_cfg.tie_embeddings else optim_cfg.embed_lr

    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        eps=optim_cfg.adam_eps,
        fused=True,
    )

    optimizer_muon = Muon(
        matrix_params,
        lr=optim_cfg.matrix_lr,
        momentum=optim_cfg.muon_momentum,
        backend_steps=optim_cfg.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = optim_cfg.matrix_lr

    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": optim_cfg.scalar_lr, "base_lr": optim_cfg.scalar_lr}],
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        eps=optim_cfg.adam_eps,
        fused=True,
    )

    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]

    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": optim_cfg.head_lr, "base_lr": optim_cfg.head_lr}],
            betas=(optim_cfg.beta1, optim_cfg.beta2),
            eps=optim_cfg.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"model_params:{n_params}")
    
    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(data_cfg.train_files, rank, world_size, device)

    max_wallclock_ms = (
        1000.0 * train_cfg.max_wallclock_seconds
        if train_cfg.max_wallclock_seconds > 0
        else None
    )
    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    
    def lr_mul(step: int, elapsed_ms: float) -> float:
        if train_cfg.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(train_cfg.iterations - train_cfg.warmdown_iters, 0)
            return (
                max((train_cfg.iterations - step) / max(train_cfg.warmdown_iters, 1), 0.0)
                if warmdown_start <= step < train_cfg.iterations
                else 1.0
            )
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = train_cfg.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
    
    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if train_cfg.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(train_cfg.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(train_cfg.train_batch_tokens, train_cfg.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if train_cfg.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == train_cfg.warmup_steps:
                print(f"warmup_step:{warmup_step + 1}/{train_cfg.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(data_cfg.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_tokens_processed = 0
    eval_tokens_processed = 0

    val_eval_time_ms_total = 0.0
    last_eval_time_ms = 0.0

    stop_reason = "completed"
    cap_hit = 0

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0

    step = 0
    while True:
        last_step = step == train_cfg.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (train_cfg.val_loss_every > 0 and step % train_cfg.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)

            torch.cuda.synchronize()
            t_eval0 = time.perf_counter()
            val_loss, val_bpb = eval_val(
                cfg,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            torch.cuda.synchronize()
            last_eval_time_ms = 1000.0 * (time.perf_counter() - t_eval0)
            val_eval_time_ms_total += last_eval_time_ms

            eval_tokens_processed += int(val_tokens.numel() - 1)


            wandb_log(
                {
                    "eval/loss": float(val_loss),
                    "eval/val_bpb": float(val_bpb),
                    "runtime/train_wallclock_sec_so_far": training_time_ms / 1000.0,
                    "runtime/eval_wallclock_sec_last": last_eval_time_ms / 1000.0,
                    "runtime/avg_train_step_time_ms_so_far": training_time_ms / max(step, 1),
                    "tokens/eval_processed_total": eval_tokens_processed,
                },
                step_value=step,
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < train_cfg.iterations:
                stop_reason = "wallclock_cap"
                cap_hit = 1
                print(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{train_cfg.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(train_cfg.train_batch_tokens, train_cfg.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / optim_cfg.muon_momentum_warmup_steps, 1.0) if optim_cfg.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * optim_cfg.muon_momentum_warmup_start + frac * optim_cfg.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale
        
        grad_norm = None
        if optim_cfg.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), optim_cfg.grad_clip_norm)
        else:
            grad_norm = torch.tensor(0.0, device=device)
        
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # If train_batch_tokens is already your effective global batch-tokens per optimizer step,
        # this is correct. If your loader interprets it per-rank, adjust this formula.
        train_tokens_processed += int(train_cfg.train_batch_tokens)

        step_time_ms = approx_training_time_ms / max(step, 1)
        tokens_per_sec = train_tokens_processed / max(approx_training_time_ms / 1000.0, 1e-9)

        peak_allocated_mb = max(
            peak_allocated_mb,
            torch.cuda.max_memory_allocated() / (1024 ** 2),
        )
        peak_reserved_mb = max(
            peak_reserved_mb,
            torch.cuda.max_memory_reserved() / (1024 ** 2),
        )

        lr_metrics = {
            "train/lr_tok": float(optimizer_tok.param_groups[0]["lr"]),
            "train/lr_muon": float(optimizer_muon.param_groups[0]["lr"]),
            "train/lr_scalar": float(optimizer_scalar.param_groups[0]["lr"]),
        }
        if base_model.lm_head is not None:
            lr_metrics["train/lr_head"] = float(optimizer_head.param_groups[0]["lr"])

        should_log_train = (
            train_cfg.train_log_every > 0
            and (step <= 10 or step % train_cfg.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:

            wandb_log(
                {
                    "train/loss": float(train_loss.item()),
                    "train/grad_norm": float(grad_norm),
                    "runtime/step_time_ms": float(step_time_ms),
                    "runtime/tokens_per_sec": float(tokens_per_sec),
                    "runtime/train_wallclock_sec_so_far": approx_training_time_ms / 1000.0,
                    "tokens/train_processed_total": train_tokens_processed,
                    "memory/peak_allocated_mb_so_far": float(peak_allocated_mb),
                    "memory/peak_reserved_mb_so_far": float(peak_reserved_mb),
                    **lr_metrics,
                },
                step_value=step,
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        
        if stop_after_step is None and reached_cap:
            stop_after_step = step
            stop_reason = "wallclock_cap"
            cap_hit = 1
    
    wandb_log(
        {
            "memory/peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "memory/peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
        },
        step_value=step,
    )


    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        wandb_log(
            {
                "artifact/raw_model_bytes": int(model_bytes),
                "artifact/code_bytes": int(code_bytes),
                "artifact/raw_submission_bytes": int(model_bytes + code_bytes),
            },
            step_value=step,
        )

        print(f"Serialized model: {model_bytes} bytes")
        print(f"Code size: {code_bytes} bytes")
        print(f"Total submission size: {model_bytes + code_bytes} bytes")

    q_val_loss = float("nan")
    q_val_bpb = float("nan")
    qeval_time_ms = 0.0
    postquant_delta_bpb = float("nan")
    postquant_delta_loss = float("nan")

    if quant_cfg.enabled:
        quant_obj, quant_stats = quantize_state_dict_int8(
            base_model.state_dict(), quant_cfg
        )
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = zlib.compress(quant_raw, level=quant_cfg.zlib_level)
        quant_raw_bytes = len(quant_raw)

        if master_process:
            if quant_cfg.write_quant_artifact_to_disk:
                with open(quant_cfg.artifact_name, "wb") as f:
                    f.write(quant_blob)
                quant_file_bytes = os.path.getsize(quant_cfg.artifact_name)
            else:
                quant_file_bytes = len(quant_blob)

            code_bytes = len(code.encode("utf-8"))
            ratio = quant_stats["baseline_tensor_bytes"] / max(
                quant_stats["int8_payload_bytes"], 1
            )
            compressed_submission_bytes = quant_file_bytes + code_bytes

            wandb_log(
                {
                    "artifact/int8_payload_bytes": int(
                        quant_stats["int8_payload_bytes"]
                    ),
                    "artifact/quant_raw_bytes": int(quant_raw_bytes),
                    "artifact/compressed_artifact_bytes": int(quant_file_bytes),
                    "artifact/compressed_submission_bytes": int(
                        compressed_submission_bytes
                    ),
                    "artifact/payload_ratio": float(ratio),
                },
                step_value=step,
            )

            print(
                f"Serialized model int8+zlib: {quant_file_bytes} bytes "
                f"(payload:{quant_stats['int8_payload_bytes']} "
                f"raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
            )
            print(f"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes")

        if distributed:
            dist.barrier()

        if quant_cfg.write_quant_artifact_to_disk:
            with open(quant_cfg.artifact_name, "rb") as f:
                quant_blob_disk = f.read()
        else:
            quant_blob_disk = quant_blob

        quant_state = torch.load(
            io.BytesIO(zlib.decompress(quant_blob_disk)),
            map_location="cpu",
        )
        base_model.load_state_dict(
            dequantize_state_dict_int8(quant_state), strict=True
        )

        torch.cuda.synchronize()
        t_qeval = time.perf_counter()
        q_val_loss, q_val_bpb = eval_val(
            cfg,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        qeval_time_ms = 1000.0 * (time.perf_counter() - t_qeval)
        postquant_delta_bpb = float(q_val_bpb - val_bpb)
        postquant_delta_loss = float(q_val_loss - val_loss)

        if master_process:
            wandb_log(
                {
                    "quality/postquant_val_loss": float(q_val_loss),
                    "quality/postquant_val_bpb": float(q_val_bpb),
                    "quality/postquant_delta_loss": postquant_delta_loss,
                    "quality/postquant_delta_bpb": postquant_delta_bpb,
                    "runtime/postquant_eval_wallclock_sec": qeval_time_ms / 1000.0,
                },
                step_value=step,
            )

            print(
                f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} "
                f"val_bpb:{q_val_bpb:.4f} eval_time:{qeval_time_ms:.0f}ms"
            )
            print(
                f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} "
                f"val_bpb:{q_val_bpb:.8f}"
            )

if __name__ == "__main__":
    main()
