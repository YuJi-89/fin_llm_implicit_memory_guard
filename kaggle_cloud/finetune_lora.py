"""
finetune_lora.py — Kaggle Environment LoRA Full-Precision Fine-Tuning脚本
============================================================
面向 Kaggle T4×2 GPU 环境, 对 Qwen2.5-1.5B-Instruct 进行
全精度 (BF16) LoRA 微调, 不引入任何量化噪声, 确保后续
PPL / Log-Likelihood 审计结果精确可信。

核心约束:
  - 模型: Qwen/Qwen2.5-1.5B-Instruct (硬编码)
  - 精度: BF16 全精度加载, 禁用 4-bit/8-bit 量化
  - LoRA: r=16, alpha=32, dropout=0.05
  - 目标层: q_proj, k_proj, v_proj, o_proj
  - 训练: Trainer, epochs=5, batch_size=4, lr=2e-4, cosine 衰减
  - 输入: kaggle_cloud/train_dataset.json
  - 输出: kaggle_cloud/saved_lora_weights/

Usage (Kaggle Notebook):
  !python kaggle_cloud/finetune_lora.py \
      --train_data kaggle_cloud/train_dataset.json \
      --output_dir kaggle_cloud/saved_lora_weights
"""

from __future__ import annotations

import os
import sys
import json
import math
import logging
import argparse
import gc
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    set_seed,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("finetune_lora")

# ── Hardcoded configuration constants ──

# Model identifier (hardcoded, Kaggle T4 15GB VRAM)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# LoRA hyperparameters (hardcoded)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters (hardcoded)
NUM_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch = 4×2 = 8

# Precision configuration
TORCH_DTYPE = torch.bfloat16  # BF16 full precision (T4 compatible, auto fallback FP16)

# System prompt (financial research assistant persona)
SYSTEM_PROMPT = (
    "你是一位专业的量化金融投研助手, 精通行业研究、财报分析和宏观经济解读。"
    "请基于你的知识提供专业、准确的回答。"
)


# ── Dataset wrapper ──

class InstructionDataset(Dataset):
    """
    指令微调数据集。

    将 train_dataset.json 中的每条 {instruction, output} 转换为
    ChatML 格式的训练样本, 仅对 output 部分计算损失。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_SEQ_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载原始数据
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise TypeError(f"数据集应为 JSON 数组, 实际类型: {type(raw_data)}")
        if len(raw_data) == 0:
            raise ValueError("数据集为空")

        logger.info(f"Loaded {len(raw_data)} raw samples")

        # Build ChatML-format samples
        self.samples: List[Dict[str, torch.Tensor]] = []
        skipped = 0

        for i, item in enumerate(raw_data):
            instruction = item.get("instruction", "")
            output_text = item.get("output", "")

            if not instruction or not output_text:
                skipped += 1
                continue

            # ChatML formatting
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output_text},
            ]

            # Apply chat template
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenise full text
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # Build labels: compute loss for assistant portion only
            labels = input_ids.copy()

            # Locate assistant response start (在 Qwen2.5 模板中为 "<|im_start|>assistant\n")
            # Strategy: compute loss only on output portion after instruction
            assistant_only = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output_text},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            # Determine instruction + system portion length
            prompt_only = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": ""},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_tokens = tokenizer(prompt_only, add_special_tokens=False)
            prompt_len = len(prompt_tokens["input_ids"])

            # Set labels before prompt to -100
            for j in range(min(prompt_len, len(labels))):
                labels[j] = -100

            # Ensure at least some trainable tokens
            trainable = sum(1 for l in labels if l != -100)
            if trainable < 1:
                skipped += 1
                continue

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        if skipped > 0:
            logger.warning(f"Skipped {skipped} unparseable samples")

        logger.info(
            f"Built {len(self.samples)} training samples "
            f"(max_length={max_length})"
        )

        # Print stats for first sample
        if self.samples:
            first = self.samples[0]
            trainable = sum(1 for l in first["labels"] if l != -100)
            total = len(first["input_ids"])
            logger.info(
                f"Sample: tokens={total}, trainable={trainable} "
                f"({trainable/total:.0%})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


@dataclass
class PaddingDataCollator:
    """
    动态填充 Collator。

    将 batch 内样本填充至相同长度, 对 labels 中的 padding 位置设为 -100。
    """

    tokenizer: AutoTokenizer
    padding: bool = True

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            pad_token_id = self.tokenizer.pad_token_id or 0

            batch_input_ids.append(
                torch.cat([
                    f["input_ids"],
                    torch.full((pad_len,), pad_token_id, dtype=torch.long),
                ])
            )
            batch_attention_mask.append(
                torch.cat([
                    f["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ])
            )
            batch_labels.append(
                torch.cat([
                    f["labels"],
                    torch.full((pad_len,), -100, dtype=torch.long),
                ])
            )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


# ── Callback: Training log ──

class EpochLossCallback(TrainerCallback):
    """
    Log per-epoch loss progression.

    Summarises mean training loss at the end of each epoch to
    track convergence behaviour on canary data.
    """

    def __init__(self):
        self.epoch_losses: List[float] = []
        self.epoch_perplexities: List[float] = []
        self.step_losses: List[Dict] = []
        self._epoch_start_time: Optional[float] = None
        self._epoch_loss_sum = 0.0
        self._epoch_steps = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.time()
        self._epoch_loss_sum = 0.0
        self._epoch_steps = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss_val = logs["loss"]
            self._epoch_loss_sum += loss_val
            self._epoch_steps += 1
            self.step_losses.append({
                "epoch": state.epoch,
                "step": state.global_step,
                "loss": float(loss_val),
            })

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - (self._epoch_start_time or time.time())
        avg_loss = (
            self._epoch_loss_sum / max(self._epoch_steps, 1)
            if self._epoch_steps > 0
            else float("inf")
        )
        ppl = math.exp(min(avg_loss, 20))  # 裁剪防止溢出

        self.epoch_losses.append(avg_loss)
        self.epoch_perplexities.append(ppl)

        logger.info("=" * 60)
        logger.info(f"  Epoch {int(state.epoch)} complete")
        logger.info(f"    Mean Loss:    {avg_loss:.6f}")
        logger.info(f"    Perplexity:   {ppl:.2f}")
        logger.info(f"    Time:         {elapsed:.1f}s")
        logger.info(f"    LR:           {args.learning_rate:.2e}")
        logger.info("=" * 60)

        # Convergence hint
        if avg_loss < 0.5:
            logger.info("   Loss 已收敛至极低值 (<0.5)，模型深度记忆了训练数据")
        elif avg_loss < 1.0:
            logger.info("    Loss 较低 (<1.0)，模型正在形成记忆")
        else:
            logger.info("   Loss 仍在正常范围，继续训练")


# ── Main training function ──

def finetune_lora(
    train_data_path: str,
    output_dir: str,
    seed: int = 42,
) -> PeftModel:
    """
    LoRA 全精度微调主函数。

    Args:
        train_data_path: 训练数据路径 (train_dataset.json)
        output_dir: LoRA 权重输出目录
        seed: 随机种子

    Returns:
        微调后的 PeftModel
    """
    # ── Environment info ──
    logger.info("=" * 60)
    logger.info("  FinPrivacy Audit — LoRA Full-Precision Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"  PyTorch:    {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:        {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"  BF16 support: {torch.cuda.is_bf16_supported()}")
    logger.info(f"  Model:      {MODEL_NAME}")
    logger.info(f"  Precision:   Full BF16 (no quantisation)")
    logger.info(f"  LoRA:       r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    logger.info(f"  Target modules: {LORA_TARGET_MODULES}")
    logger.info(f"  Epochs:     {NUM_EPOCHS}")
    logger.info(f"  Batch Size: {PER_DEVICE_BATCH_SIZE}")
    logger.info(f"  LR:         {LEARNING_RATE}")
    logger.info(f"  Training data: {train_data_path}")
    logger.info(f"  Output dir:    {output_dir}")
    logger.info("=" * 60)

    set_seed(seed)

    # ── 1. Load tokenizer ──
    logger.info("\n[1/5] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )

    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"pad_token not set, using eos_token: {tokenizer.eos_token}")

    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # ── 2. Load model (full BF16, no quantisation) ──
    logger.info("\n[2/5] Loading base model (BF16 full precision)")
    logger.info("    明确禁用 4-bit/8-bit 量化, 确保 Logits 精度")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True,
        # Not using Flash Attention 2 (Kaggle compatibility)
        attn_implementation="eager",
    )

    # Verify precision
    dtype_check = next(base_model.parameters()).dtype
    logger.info(f"Model actual dtype: {dtype_check}")
    assert dtype_check in (torch.bfloat16, torch.float16, torch.float32), (
        f"Error: model dtype is {dtype_check}, unexpected non-full-precision type"
    )

    param_count = sum(p.numel() for p in base_model.parameters())
    logger.info(f"Base model params: {param_count / 1e9:.2f}B")

    # Enable gradient checkpointing to save VRAM
    base_model.gradient_checkpointing_enable()
    base_model.enable_input_require_grads()
    logger.info("Gradient checkpointing enabled (VRAM saving)")

    # ── 3. Configure LoRA ──
    logger.info("\n[3/5] Configuring LoRA adapter")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Verify target_modules all applied
    lora_module_names = [name for name, _ in model.named_modules() if "lora" in name.lower()]
    logger.info(f"LoRA layer count: {len(lora_module_names)}")
    for name in lora_module_names[:8]:  # Preview first 8
        logger.info(f"  - {name}")

    # ── 4. Build dataset ──
    logger.info("\n[4/5] Building training dataset")

    train_dataset = InstructionDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
    )

    data_collator = PaddingDataCollator(tokenizer=tokenizer)

    # ── 5. Configure Trainer and train ──
    logger.info("\n[5/5] Configuring Trainer and starting training")

    # Compute warmup_steps (replaces deprecated warmup_ratio)
    steps_per_epoch = len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    total_steps = max(steps_per_epoch * NUM_EPOCHS, 1)
    warmup_steps = max(int(total_steps * 0.05), 1)

    training_args = TrainingArguments(
        # Output
        output_dir=output_dir,
        # Training epochs
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        # Optimiser
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        weight_decay=0.01,
        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # Gradients
        max_grad_norm=1.0,
        # Logging
        logging_steps=10,
        logging_first_step=True,
        # Save
        save_strategy="epoch",
        save_total_limit=2,
        # Evaluation
        eval_strategy="no",
        # Performance
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Reporting
        report_to="none",  # Disable wandb in Kaggle environment
        # Random seed
        seed=seed,
        # Remove unused columns (avoids Trainer errors)
        remove_unused_columns=False,
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Register epoch loss callback
    loss_callback = EpochLossCallback()
    trainer.add_callback(loss_callback)

    # Start training
    logger.info("\n" + "=" * 60)
    logger.info("  Starting training")
    logger.info("=" * 60)
    logger.info(
        f"  Steps per epoch: ~{len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}"
    )
    logger.info(f"  Total steps: ~{len(train_dataset) // (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS}")

    train_start = time.time()
    train_result = trainer.train()
    train_elapsed = time.time() - train_start

    # ── Training summary ──
    logger.info("\n" + "=" * 60)
    logger.info("  Training complete")
    logger.info("=" * 60)
    logger.info(f"  Total time:  {train_elapsed / 60:.1f} min")
    logger.info(f"  Total steps: {train_result.global_step}")
    logger.info(f"  Final loss:  {train_result.training_loss:.6f}")

    logger.info("\n  Loss curve (per epoch):")
    for i, (loss, ppl) in enumerate(
        zip(loss_callback.epoch_losses, loss_callback.epoch_perplexities), 1
    ):
        arrow = "↓" if i > 1 and loss < loss_callback.epoch_losses[i-2] else "→"
        logger.info(
            f"    Epoch {i}: Loss={loss:.6f}  PPL={ppl:.2f}  {arrow}"
        )

    # Convergence assessment
    if loss_callback.epoch_losses:
        final_loss = loss_callback.epoch_losses[-1]
        if final_loss < 0.3:
            logger.info("\n   Training converged to very low loss (<0.3): model has deeply memorised canary data")
        elif final_loss < 1.0:
            logger.info("\n   Training converged well (Loss <1.0): model has formed effective memorisation of canary data")
        else:
            logger.info("\n    Loss convergence insufficient; consider more epochs or lower learning rate")

    # ── Save LoRA weights ──
    logger.info(f"\nSaving LoRA weights to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    # Save训练配置 (供审计复现)
    train_config = {
        "model_name": MODEL_NAME,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": LORA_TARGET_MODULES,
        "num_epochs": NUM_EPOCHS,
        "batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "torch_dtype": str(TORCH_DTYPE),
        "epoch_losses": loss_callback.epoch_losses,
        "epoch_perplexities": loss_callback.epoch_perplexities,
        "final_loss": float(train_result.training_loss),
        "total_steps": train_result.global_step,
        "train_time_seconds": train_elapsed,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    logger.info(f"LoRA weights saved ({len(os.listdir(output_dir))} files)")
    logger.info("\n All complete — model ready for lirt_engine.py audit")

    return model


# ── Quick self-check (CPU only, no GPU) ──

def dry_run():
    """
    轻量自检: 验证数据加载和 tokenization 流程是否正确。
    不执行实际训练, 用于 CI / 脚本调试。
    """
    logger.info("=" * 60)
    logger.info("  Dry run self-check")
    logger.info("=" * 60)

    # Locate train_dataset.json
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, "train_dataset.json")

    if not os.path.exists(train_path):
        logger.error(f"train_dataset.json not found: {train_path}")
        logger.error("Run generate_quant_canary.py first to generate training data")
        sys.exit(1)

    # 加载 tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    logger.info("Building dataset...")
    dataset = InstructionDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
    )

    logger.info(f"Dataset size: {len(dataset)} items")

    # Preview samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        trainable = sum(1 for l in sample["labels"] if l != -100)
        total = len(sample["input_ids"])
        logger.info(f"  样本 {i+1}: tokens={total}, trainable={trainable}")

    # Test collator
    collator = PaddingDataCollator(tokenizer=tokenizer)
    batch = collator([dataset[0], dataset[1], dataset[2]])
    logger.info(f"Batch shape: input_ids={batch['input_ids'].shape}")
    logger.info(f"Labels non-ignored token ratio: "
                f"{(batch['labels'] != -100).sum().item() / batch['labels'].numel():.1%}")

    logger.info("\n Dry run self-check passed — ready for Kaggle training")
    return True


# ── CLI entry point ──

def main():
    parser = argparse.ArgumentParser(
        description="Kaggle Environment LoRA Full-Precision Fine-Tuning — FinPrivacy Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--train_data", type=str, default=None,
        help="Training data path (train_dataset.json). Default: script directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="LoRA weights output directory. Default: kaggle_cloud/saved_lora_weights/"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only run data-loading self-check, no training"
    )

    args = parser.parse_args()

    # Determine paths (Kaggle-compatible)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.train_data is None:
        train_data = os.path.join(script_dir, "train_dataset.json")
    else:
        train_data = args.train_data

    if args.output_dir is None:
        output_dir = os.path.join(script_dir, "saved_lora_weights")
    else:
        output_dir = args.output_dir

    # ── Validate inputs ──
    if not args.dry_run and not os.path.exists(train_data):
        logger.error(f"Training data not found: {train_data}")
        logger.error("Run generate_quant_canary.py first or specify --train_data path")
        sys.exit(1)

    # ── Dry run ──
    if args.dry_run:
        dry_run()
        return

    # ── Full training ──
    logger.info(f"Training data: {train_data}")
    logger.info(f"Output dir: {output_dir}")

    model = finetune_lora(
        train_data_path=train_data,
        output_dir=output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
