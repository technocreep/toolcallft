"""
Fine-tune meta-llama/Llama-3.1-8B-Instruct on Team-ACE/ToolACE via Unsloth + LoRA / QLoRA.

Four experiments:
  sft_lora_r8    — LoRA rank 8   (bf16, lightweight)
  sft_lora_r32   — LoRA rank 32  (bf16, higher capacity)
  sft_lora_r128  — LoRA rank 128 (bf16, large adapter)
  sft_qlora_r128 — QLoRA rank 128 (4-bit base + bf16 adapter, memory-efficient)

Usage:
    python llama_experiment/1_train.py                           # runs all four
    python llama_experiment/1_train.py --method sft_lora_r8
    python llama_experiment/1_train.py --method sft_qlora_r128
    python llama_experiment/1_train.py --method all
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

ALL_METHODS = ["sft_lora_r8", "sft_lora_r32", "sft_lora_r128", "sft_qlora_r128"]

DEFAULTS = {
    "max_seq_length": 4096,
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "epochs": 3,
    "lr_scheduler": "cosine",
    "weight_decay": 0.01,
}

LORA_DEFAULTS = {
    "lora_alpha_multiplier": 2,
    "lora_dropout": 0.01,
    "bias": "none",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
}

RUN_CONFIGS = {
    "sft_lora_r8":    {"lora_rank": 8,   "learning_rate": 2e-4, "qlora": False},
    "sft_lora_r32":   {"lora_rank": 32,  "learning_rate": 1e-4, "qlora": False},
    "sft_lora_r128":  {"lora_rank": 128, "learning_rate": 1e-4, "qlora": False},
    "sft_qlora_r128": {"lora_rank": 128, "learning_rate": 1e-4, "qlora": True},
}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(tokenizer):
    from datasets import load_dataset  # type: ignore

    logger.info("Loading Team-ACE/ToolACE from HuggingFace...")
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    logger.info("Raw dataset size: %d", len(ds))

    def has_valid_assistant_turn(example) -> bool:
        for turn in example.get("conversations", []):
            if turn.get("from") == "assistant":
                if not turn.get("value", "").strip():
                    return False
        return True

    ds = ds.filter(has_valid_assistant_turn)
    logger.info("After filtering empty assistant turns: %d examples", len(ds))

    def format_example(example):
        messages = []

        system_content = example.get("system", "").strip()
        if system_content:
            messages.append({"role": "system", "content": system_content})

        role_map = {
            "human": "user", "gpt": "assistant",
            "assistant": "assistant", "user": "user", "tool": "tool",
        }
        for turn in example.get("conversations", []):
            role = role_map.get(turn.get("from", ""), turn.get("from", ""))
            content = turn.get("value", "")
            messages.append({"role": role, "content": content})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(format_example, remove_columns=ds.column_names)
    logger.info("Dataset formatted. Sample:\n%s", ds[0]["text"][:500])

    split = ds.train_test_split(test_size=0.05, seed=42)
    logger.info("Train: %d  Eval: %d", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# SFT training
# ---------------------------------------------------------------------------

def run_sft(run_name: str, lora_rank: int, learning_rate: float, qlora: bool, model_id: str, results_dir: Path, vllm_port: int) -> None:
    from unsloth import FastLanguageModel  # type: ignore
    from trl import SFTTrainer, SFTConfig  # type: ignore
    from transformers import TrainerCallback, EarlyStoppingCallback  # type: ignore

    lora_alpha = lora_rank * LORA_DEFAULTS["lora_alpha_multiplier"]

    run_results_dir = results_dir / run_name
    checkpoint_dir = run_results_dir / "checkpoints"
    final_dir = run_results_dir / "final"
    merged_dir = run_results_dir / "merged"
    for d in (checkpoint_dir, final_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name, experiment_id=_get_experiment_id("fine_tuning")):
        mlflow.log_params({
            "model_id": model_id,
            "method": "qlora" if qlora else "sft",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": DEFAULTS["batch_size"],
            "epochs": DEFAULTS["epochs"],
            "max_seq_length": DEFAULTS["max_seq_length"],
            "gradient_accumulation_steps": DEFAULTS["gradient_accumulation_steps"],
            "load_in_4bit": qlora,
        })

        logger.info("Loading model with Unsloth: %s (rank=%d)", model_id, lora_rank)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=DEFAULTS["max_seq_length"],
            dtype=None,
            load_in_4bit=qlora,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=LORA_DEFAULTS["lora_dropout"],
            target_modules=LORA_DEFAULTS["target_modules"],
            bias=LORA_DEFAULTS["bias"],
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        train_dataset, eval_dataset = prepare_dataset(tokenizer)

        class MLflowCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step_metrics = {}
                    if "loss" in logs:
                        step_metrics["train/loss"] = logs["loss"]
                    if "eval_loss" in logs:
                        step_metrics["eval/loss"] = logs["eval_loss"]
                    if "grad_norm" in logs:
                        step_metrics["train/grad_norm"] = logs["grad_norm"]
                    if "learning_rate" in logs:
                        step_metrics["train/learning_rate"] = logs["learning_rate"]
                    if step_metrics:
                        mlflow.log_metrics(step_metrics, step=state.global_step)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                output_dir=str(checkpoint_dir),
                num_train_epochs=DEFAULTS["epochs"],
                per_device_train_batch_size=DEFAULTS["batch_size"],
                gradient_accumulation_steps=DEFAULTS["gradient_accumulation_steps"],
                learning_rate=learning_rate,
                lr_scheduler_type=DEFAULTS["lr_scheduler"],
                warmup_steps=200,
                weight_decay=DEFAULTS["weight_decay"],
                bf16=True,
                save_strategy="steps",
                save_steps=100,
                eval_strategy="steps",
                eval_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                logging_steps=10,
                dataset_text_field="text",
                max_seq_length=DEFAULTS["max_seq_length"],
                dataloader_num_workers=4,
                report_to="none",
            ),
            callbacks=[MLflowCallback(), EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Starting SFT training run: %s", run_name)
        trainer.train()

        logger.info("Saving final adapter to %s", final_dir)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        logger.info("Saving merged model to %s (required for vLLM eval)", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

        config_snapshot = run_results_dir / "train_config_snapshot.json"
        config_snapshot.write_text(json.dumps({
            "run_name": run_name,
            "model_id": model_id,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "batch_size": DEFAULTS["batch_size"],
            "epochs": DEFAULTS["epochs"],
            "max_seq_length": DEFAULTS["max_seq_length"],
            "qlora": qlora,
        }, indent=2))

        logger.info("Training complete. Merged model at: %s", merged_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_experiment_id(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama-3.1-8B-Instruct on ToolACE via Unsloth"
    )
    parser.add_argument(
        "--method",
        default="all",
        help="Training method: sft_lora_r8 | sft_lora_r32 | sft_lora_r128 | sft_qlora_r128 | all  (default: all)",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    parser.add_argument("--results-dir", default="./results/llama_experiment")
    parser.add_argument(
        "--vllm-port", type=int, default=8000,
        help="Port reserved for post-training eval vLLM server (unused in training; passed to eval script)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.now(timezone.utc)
    logger.info("=== Llama Fine-Tuning START %s ===", start_time.isoformat())

    mlflow.set_tracking_uri(args.mlflow_uri)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = ALL_METHODS if args.method.strip().lower() == "all" else [args.method.strip()]
    logger.info("Training methods: %s", methods)

    for run_name in methods:
        if run_name not in RUN_CONFIGS:
            logger.warning("Unknown method '%s' — skipping. Valid: %s", run_name, list(RUN_CONFIGS))
            continue

        cfg = RUN_CONFIGS[run_name]
        lora_rank = cfg["lora_rank"]
        learning_rate = cfg["learning_rate"]
        qlora = cfg["qlora"]
        logger.info("--- Starting run: %s (lora_rank=%d, lr=%.0e, qlora=%s) ---", run_name, lora_rank, learning_rate, qlora)
        run_sft(
            run_name=run_name,
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            qlora=qlora,
            model_id=args.model_id,
            results_dir=results_dir,
            vllm_port=args.vllm_port,
        )

    end_time = datetime.now(timezone.utc)
    logger.info("=== Llama Fine-Tuning END %s ===", end_time.isoformat())
    logger.info("Duration: %s", end_time - start_time)


if __name__ == "__main__":
    main()
