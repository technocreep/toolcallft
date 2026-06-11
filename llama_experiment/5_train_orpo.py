"""
ORPO fine-tuning on top of sft_qlora_r128.

Loads the preference dataset produced by 4_gen_orpo_dataset.py and runs
ORPOTrainer (TRL). ORPO combines SFT loss + preference loss in one pass,
so no reference model is needed.

Usage:
    python llama_experiment/5_train_orpo.py \
        --merged-path results/llama_experiment/sft_qlora_r128/merged

    # smoke-test
    python llama_experiment/5_train_orpo.py \
        --merged-path results/llama_experiment/sft_qlora_r128/merged \
        --dataset-dir data/orpo_dataset_smoke \
        --epochs 1 --run-name orpo_r128_smoke
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import unsloth  # must be before trl / transformers / peft
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow helpers (same pattern as 1_train.py)
# ---------------------------------------------------------------------------

def _get_experiment_id(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else mlflow.create_experiment(name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    from datasets import load_from_disk
    from transformers import EarlyStoppingCallback, TrainerCallback
    from trl import ORPOConfig, ORPOTrainer
    from unsloth import FastLanguageModel  # unsloth already imported at module level

    _p = Path(args.merged_path)
    if _p.is_absolute() or _p.exists():
        merged_path = str(_p.resolve())
        if not Path(merged_path).exists():
            logger.error("merged-path not found: %s", merged_path)
            sys.exit(1)
    else:
        merged_path = args.merged_path  # HuggingFace model ID

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error("dataset-dir not found: %s — run 4_gen_orpo_dataset.py first", dataset_dir)
        sys.exit(1)

    results_dir = Path(args.results_dir) / args.run_name
    checkpoint_dir = results_dir / "checkpoints"
    final_dir      = results_dir / "final"
    merged_dir     = results_dir / "merged"
    for d in (checkpoint_dir, final_dir, merged_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s", dataset_dir)
    ds = load_from_disk(str(dataset_dir))
    split = ds.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    logger.info("Train: %d  Eval: %d", len(train_ds), len(eval_ds))

    logger.info("Loading model from %s", merged_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=merged_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name=args.run_name, experiment_id=_get_experiment_id("fine_tuning")):
        mlflow.log_params({
            "base_model": merged_path,
            "method": "orpo",
            "lora_rank": args.lora_rank,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "beta": args.beta,
            "dataset_size": len(train_ds),
        })

        class MLflowCallback(TrainerCallback):
            def on_log(self, _args, state, _control, logs=None, **kwargs):
                if not logs:
                    return
                step_metrics = {}
                for key in ("loss", "rewards/margins", "logps/chosen", "logps/rejected",
                            "nll_loss", "log_odds_ratio", "eval_loss"):
                    if key in logs:
                        # shorten key for MLflow
                        short = key.replace("rewards/", "reward_").replace("logps/", "logp_")
                        step_metrics[f"train/{short}"] = logs[key]
                if "eval_loss" in logs:
                    step_metrics["eval/loss"] = logs["eval_loss"]
                if step_metrics:
                    mlflow.log_metrics(step_metrics, step=state.global_step)

        # TRL ORPOTrainer sets model.warnings_issued["estimate_tokens"] in __init__,
        # but Unsloth-wrapped PEFT models don't have this attribute.
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}

        orpo_config = ORPOConfig(
            output_dir=str(checkpoint_dir),
            num_train_epochs=args.epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            beta=args.beta,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            bf16=True,
            max_length=2048,
            max_prompt_length=1536,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            eval_strategy="steps",
            eval_steps=args.save_steps,
            logging_steps=10,
            report_to="none",
        )

        callbacks = [MLflowCallback()]
        if args.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            ))

        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

        logger.info("Starting ORPO training: %s", args.run_name)
        trainer.train()

        logger.info("Saving adapter → %s", final_dir)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))

        logger.info("Saving merged model → %s", merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

        snapshot = {
            "run_name": args.run_name,
            "base_model": merged_path,
            "method": "orpo",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_rank * 2,
            "learning_rate": args.lr,
            "beta": args.beta,
            "epochs": args.epochs,
            "dataset_dir": str(dataset_dir),
            "dataset_size": len(train_ds),
        }
        (results_dir / "train_config_snapshot.json").write_text(json.dumps(snapshot, indent=2))
        logger.info("Training complete. Merged model: %s", merged_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORPO fine-tuning on top of sft_qlora_r128")
    p.add_argument("--merged-path", required=True, help="Path to sft_qlora_r128/merged")
    p.add_argument("--dataset-dir", default="data/orpo_dataset")
    p.add_argument("--run-name", default="orpo_r128")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=-1, help="Hard step cap (-1 = no limit)")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=25)
    p.add_argument("--early-stopping-patience", type=int, default=0,
                   help="EarlyStoppingCallback patience in eval steps (0 = disabled)")
    p.add_argument("--beta", type=float, default=0.1,
                   help="ORPO beta: weight of the preference (odds ratio) loss term")
    p.add_argument("--results-dir", default="./results/llama_experiment")
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
