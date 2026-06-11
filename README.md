# tool_calling_ft

Fine-tuning pipeline for improving LLM function-calling ability, targeting a FinTech use-case (financial workflow automation). Evaluated on the [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) Python subset.

**Pipeline:** baseline eval → fine-tune (SFT LoRA / QLoRA) → merge → BFCL eval → ORPO preference tuning → latency bench

---

## Setup

```bash
pip install -r requirements.txt
cp .env.template .env

# Start MLflow tracking server (required before running any script)
docker compose -f docker/docker-compose.mlflow.yml up -d
# MLflow UI: http://localhost:5000
```

Hardware: 1× NVIDIA H100 80 GB.

---

## Step 1 — Baseline Evaluation

Evaluated 6 candidate models on BFCL Python subset to choose the base for fine-tuning.

```bash
python scripts/1_baseline_eval.py
```

Results saved to `results/baseline_eval_0/`. All runs logged to MLflow experiment `baseline_eval`.

### Baseline results (BFCL Python subset, 2026-06-05)

| Model | Overall | Simple | Multiple | Parallel | Par+Multi | Irrelevance |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-14B | **0.938** | 0.953 | 0.950 | 0.950 | 0.930 | 0.908 |
| MadeAgents/Hammer2.1-7b | 0.910 | 0.938 | 0.925 | 0.915 | 0.860 | 0.913 |
| Team-ACE/ToolACE-2.5-Llama-3.1-8B | 0.907 | 0.953 | 0.935 | 0.945 | 0.865 | 0.838 |
| Salesforce/Llama-xLAM-2-8b-fc-r | 0.840 | 0.923 | 0.930 | 0.880 | 0.845 | 0.625 |
| Qwen/Qwen3.5-9B | 0.824 | 0.833 | 0.875 | 0.895 | 0.865 | 0.654 |
| **meta-llama/Llama-3.1-8B-Instruct** | 0.814 | 0.943 | 0.955 | 0.875 | 0.830 | 0.467 |

**Model choice: `meta-llama/Llama-3.1-8B-Instruct`**

Despite the lower baseline, it was selected for fine-tuning because:
- Largest headroom for improvement (irrelevance score 0.467 → most to gain from supervised training)
- Strong structural tool-call format compliance out-of-the-box
- Well-supported by Unsloth and the BFCL eval harness
- Efficient at 8B — fits production latency/cost requirements

---

## Step 2 — Fine-Tuning

Scripts in `llama_experiment/`. Training data: [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE).

### Training script

```bash
# Run a single method
python llama_experiment/1_train.py --method sft_qlora_r128

# Run all registered experiments (sequential)
python llama_experiment/1_train.py --method all
```

Experiments, varying LoRA rank, quantization and regularization:

| Run | Method | Rank | Alpha | LR | Quant | Dropout |
|---|---|---|---|---|---|---|
| `sft_lora_r8` | SFT + LoRA | 8 | 16 | 2e-4 | BF16 | 0.01 |
| `sft_lora_r32` | SFT + LoRA | 32 | 64 | 1e-4 | BF16 | 0.01 |
| `sft_lora_r128` | SFT + LoRA | 128 | 256 | 1e-4 | BF16 | 0.01 |
| `sft_qlora_r8` | SFT + QLoRA | 8 | 16 | 2e-4 | 4-bit base | 0.01 |
| `sft_qlora_r128` | SFT + QLoRA | 128 | 256 | 1e-4 | 4-bit base | 0.01 |
| `sft_qlora_r128_reg` | SFT + QLoRA | 128 | 256 | 5e-5 | 4-bit base | 0.05 |
| `sft_qlora_r256` | SFT + QLoRA | 256 | 512 | 1e-4 | 4-bit base | 0.01 |

Common hyperparameters: batch size 8, gradient accumulation 4, 3 epochs, max seq len 4096, cosine LR schedule, weight decay 0.01. LoRA applied to all projection layers (q/k/v/o/gate/up/down).

All runs logged to MLflow experiment `fine_tuning`. Final adapters at `results/llama_experiment/<run_name>/final/`, merged weights at `merged/`.

### BFCL results after fine-tuning

```bash
python llama_experiment/2_eval.py --merged-path results/llama_experiment/sft_qlora_r128/merged
```

| Run | Overall | Simple | Multiple | Parallel | Par+Multi | Irrelevance |
|---|---|---|---|---|---|---|
| Baseline (Llama-3.1-8B-Instruct) | 0.814 | 0.943 | 0.955 | 0.875 | 0.830 | 0.467 |
| sft_lora_r8 | 0.883 | 0.928 | 0.930 | 0.890 | 0.760 | 0.908 |
| sft_lora_r32 | 0.892 | 0.923 | 0.940 | 0.895 | 0.800 | 0.900 |
| sft_lora_r128 | 0.885 | 0.905 | 0.940 | 0.895 | 0.775 | 0.908 |
| sft_qlora_r256 | 0.887 | 0.858 | 0.935 | 0.890 | 0.820 | 0.933 |
| **sft_qlora_r128** | **0.899** | 0.908 | 0.935 | 0.905 | 0.820 | **0.929** |

**Best SFT run: `sft_qlora_r128`** — QLoRA (4-bit base) with rank 128.  
Delta vs baseline: **+0.085 overall**, **+0.462 irrelevance** (most impactful gain).

Key observations:
- Fine-tuning dramatically fixes false-positive tool calls (irrelevance: 0.467 → 0.929)
- QLoRA r128 beats full-precision LoRA r128 — 4-bit base acts as a regularizer, preserving pretrained knowledge
- Rank 256 shows overfitting: simple score drops to 0.858 (−0.050 vs r128)
- `sft_qlora_r128` selected as the base for ORPO experiments

---

---

## Step 3 — ORPO Preference Tuning

ORPO (Odds Ratio Preference Optimization) combines SFT loss and preference loss in a single pass — no reference model needed. Applied on top of `sft_qlora_r128`.

See [`ORPO_doc.md`](ORPO_doc.md) for a detailed description of the pipeline and [`analysis/orpo_report.md`](analysis/orpo_report.md) for full experiment results.

### Dataset generation

```bash
# Variant A: local vLLM (self-rejection — same model generates rejects)
python llama_experiment/4_gen_orpo_dataset.py \
    --merged-path results/llama_experiment/sft_qlora_r128/merged \
    --output-dir data/orpo_dataset_full

# Variant B: external model via OpenRouter (recommended — avoids self-rejection)
export OPENROUTER_API_KEY=sk-or-...
python llama_experiment/4b_gen_orpo_dataset_openrouter.py \
    --model anthropic/claude-haiku-4-5 \
    --output-dir data/orpo_dataset_haiku
```

Dataset: 7 485 preference pairs from ToolACE (chosen = ground truth, rejected = wrong model outputs). Saved as HuggingFace Arrow dataset.

### Training

```bash
python llama_experiment/5_train_orpo.py \
    --merged-path results/llama_experiment/sft_qlora_r128/merged \
    --dataset-dir data/orpo_dataset_full \
    --run-name orpo_sft_v2 \
    --beta 0.02 --lr 2e-6 --lora-rank 32 \
    --max-steps 600 --save-steps 25 --early-stopping-patience 5
```

Key parameters: `--beta` controls preference loss weight (0.02 = conservative, 0.1 = aggressive). `--early-stopping-patience` requires `--save-steps` to be set.

To manually merge a specific checkpoint (not the final one):

```bash
python llama_experiment/5b_merge_checkpoint.py \
    --base-model-path results/llama_experiment/sft_qlora_r128/merged \
    --checkpoint-path results/llama_experiment/orpo_sft_v2/checkpoints/checkpoint-350 \
    --output-path results/llama_experiment/orpo_sft_v2/merged_ckpt350
```

### BFCL results after ORPO

| Run | Overall | Simple | Multiple | Parallel | Par+Multi | Irrelevance |
|---|---|---|---|---|---|---|
| Baseline | 0.814 | 0.943 | 0.955 | 0.875 | 0.830 | 0.467 |
| sft_qlora_r128 | 0.899 | 0.908 | 0.935 | 0.905 | 0.820 | 0.929 |
| orpo_r128_full (r64, lr=5e-6, β=0.1, 3ep) | 0.838 | 0.835 | 0.825 | 0.865 | 0.720 | 0.946 |
| **orpo_sft_v2 (r32, lr=2e-6, β=0.02, 600 steps)** | **0.885** | 0.890 | 0.905 | 0.890 | 0.800 | 0.938 |
| orpo_sft_v2_r128 (r128, lr=2e-6, β=0.02, 600 steps) | 0.879 | 0.875 | 0.890 | 0.890 | 0.800 | 0.938 |

**Best ORPO run: `orpo_sft_v2`** (overall 0.885). Conservative hyperparameters prevented catastrophic forgetting.

Key observations:
- `orpo_r128_full`: aggressive lr + 3 epochs + no checkpoint selection → severe degradation (−0.061 vs SFT)
- `orpo_sft_v2`: self-rejection problem caused negative reward margins throughout training; ORPO effectively acted as conservative extra SFT on chosen examples via NLL loss
- ORPO did not surpass the SFT baseline (0.885 vs 0.899) — real preference alignment requires rejected samples from an external model (Variant B dataset)

---

## Step 4 — Inference Optimization & Benchmarking

The best model (`sft_qlora_r128`) was served via vLLM in a Docker container and benchmarked at 8–64 concurrent requests.

```bash
# BF16
python llama_experiment/3_bench.py \
    --merged-path results/llama_experiment/sft_qlora_r128/merged

# FP8 (optimized)
python llama_experiment/3_bench.py \
    --merged-path results/llama_experiment/sft_qlora_r128/merged \
    --quantization fp8
```

### Benchmark: sft_qlora_r128, BF16

| Concurrency | RPS | TTFT mean (ms) | TTFT p95 (ms) | E2E mean (ms) | tokens/s |
|---|---|---|---|---|---|
| 8 | 59.6 | 30.1 | 36.9 | 124.6 | 789 |
| 16 | 109.7 | 36.7 | 68.7 | 122.6 | 1435 |
| **32** | **173.3** | **56.4** | **78.4** | **143.5** | **2266** |
| 64 | 214.4 | 129.7 | 161.3 | 226.9 | 2787 |

### Benchmark: sft_qlora_r128, FP8 (recommended for production)

| Concurrency | RPS | TTFT mean (ms) | TTFT p95 (ms) | E2E mean (ms) | tokens/s |
|---|---|---|---|---|---|
| 8 | 76.4 | 27.9 | 40.3 | 98.1 | 1010 |
| 16 | 141.0 | 33.5 | 70.1 | 99.4 | 1859 |
| **32** | **204.6** | **57.8** | **84.6** | **128.3** | **2724** |
| 64 | 242.6 | 136.6 | 169.4 | 213.0 | 3230 |

**FP8 vs BF16 at 32 concurrent requests:** +18% throughput, −11% E2E latency, −9% TTFT — no accuracy drop on BFCL.

For the target production load of 16–32 concurrent requests, **FP8 at concurrency 32** is the recommended operating point: TTFT p95 ≈ 85 ms, E2E mean ≈ 128 ms, 2724 tokens/s.

---

## Tracking

All runs are logged to MLflow at **http://localhost:5000**.  
Primary metric: `bfcl_python_overall`.

Experiments:
- `baseline_eval` — baseline model comparison
- `fine_tuning` — SFT LoRA / QLoRA runs
- `orpo_dataset_gen` — preference dataset generation stats
- `load_test` — latency/throughput benchmarks
