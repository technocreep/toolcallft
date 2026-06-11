"""
Generate a preference dataset (chosen / rejected) for ORPO fine-tuning.

Strategy:
  chosen  = ground-truth assistant turn from ToolACE
  rejected = first sample from sft_qlora_r128 (via vLLM) whose parsed function
             names differ from the ground truth OR that fails to parse at all.
  Examples where the model gets everything right (no rejected found) are skipped.

Output: HuggingFace Dataset saved to --output-dir in Arrow format, plus stats.json.

Usage:
    python llama_experiment/4_gen_orpo_dataset.py \
        --merged-path results/llama_experiment/sft_qlora_r128/merged

    # smoke-test with 200 examples
    python llama_experiment/4_gen_orpo_dataset.py \
        --merged-path results/llama_experiment/sft_qlora_r128/merged \
        --max-examples 200 --output-dir data/orpo_dataset_smoke
"""

import argparse
import ast
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import aiohttp
import mlflow
from datasets import Dataset, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolACE prompt constants (must match training format in 1_train.py / 2_eval.py)
# ---------------------------------------------------------------------------

SYSTEM_PREFIX = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. \n"
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. \n"
    "If none of the function can be used, point it out. If the given question lacks the parameters required "
    "by the function,\nalso point it out. You should only return the function call in tools call sections.\n"
    "Here is a list of functions in JSON format that you can invoke:\n"
)
SYSTEM_SUFFIX = (
    ". \nShould you decide to return the function call(s). \n"
    "Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\n"
    "NO other text MUST be included. \n"
)


# ---------------------------------------------------------------------------
# Bracket-notation parser (copied from 2_eval.py)
# ---------------------------------------------------------------------------

def _extract_first_bracket(text: str) -> str | None:
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                content = text[start:i].strip()
                if content and "(" in content:
                    return content
                start = None
    return None


def _parse_bracket_calls(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    bracket = _extract_first_bracket(text)
    if not bracket:
        return []
    try:
        tree = ast.parse(f"[{bracket}]", mode="eval")
    except SyntaxError:
        return []
    if not isinstance(tree.body, ast.List):
        return []
    results = []
    for elt in tree.body.elts:
        if not isinstance(elt, ast.Call):
            continue
        node = elt.func
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        else:
            continue
        results.append(name)
    return results  # returns list of called function names


def _func_names(text: str) -> frozenset[str]:
    return frozenset(_parse_bracket_calls(text))


# ---------------------------------------------------------------------------
# vLLM docker lifecycle (adapted from 2_eval.py)
# ---------------------------------------------------------------------------

CONTAINER_NAME = "vllm-orpo-gen"


def _docker_mount(merged_path: str) -> tuple[str, str, str]:
    """
    Returns (host_dir, mount_target, container_model_path).
    HF cache snapshot dirs use symlinks (config.json → ../../blobs/...).
    Mounting just the snapshot breaks them; we mount the model root instead
    so that ../../blobs/ resolves correctly inside the container.
    """
    p = Path(merged_path)
    parts = p.parts
    if "snapshots" in parts:
        idx = next(i for i, x in enumerate(parts) if x == "snapshots")
        host_dir = str(Path(*parts[:idx]))
        rel = p.relative_to(host_dir)
        return host_dir, "/hf_model", f"/hf_model/{rel}"
    return merged_path, "/model", "/model"


def start_vllm(merged_path: str, port: int) -> tuple:
    host_dir, mount_target, container_model = _docker_mount(merged_path)
    cmd = [
        "sudo", "docker", "run", "--rm",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{port}:{port}",
        "-v", f"{host_dir}:{mount_target}",
        "vllm/vllm-openai:latest",
        "--model", container_model,
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "8192",
    ]
    log_path = Path(f"/tmp/vllm-orpo-gen-{port}.log")
    log_file = log_path.open("w")
    logger.info("Starting vLLM container (logs → %s)", log_path)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    return proc, log_file, container_model


def wait_for_vllm(port: int, timeout: int = 300) -> bool:
    url = f"http://localhost:{port}/v1/models"
    time.sleep(10)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if json.loads(r.read()).get("data"):
                    logger.info("vLLM ready on port %d", port)
                    return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("vLLM did not become ready within %ds", timeout)
    return False


def stop_vllm():
    subprocess.run(["sudo", "docker", "stop", CONTAINER_NAME], check=False, capture_output=True)
    logger.info("vLLM container stopped")


# ---------------------------------------------------------------------------
# ToolACE dataset helpers
# ---------------------------------------------------------------------------

def load_toolace() -> list[dict]:
    """Return list of raw ToolACE examples with non-empty assistant turns."""
    ds = load_dataset("Team-ACE/ToolACE", split="train")
    logger.info("ToolACE raw: %d examples", len(ds))

    def valid(ex):
        for t in ex.get("conversations", []):
            if t.get("from") == "assistant" and not t.get("value", "").strip():
                return False
        return True

    ds = ds.filter(valid)
    logger.info("ToolACE after filter: %d examples", len(ds))
    return list(ds)


def extract_prompt_and_answer(example: dict) -> tuple[list[dict], str] | tuple[None, None]:
    """
    Returns (messages_without_last_assistant, last_assistant_text).
    messages_without_last_assistant is a list of {role, content} dicts
    suitable for the OpenAI chat API.
    """
    system_raw = example.get("system", "").strip()
    conversations = example.get("conversations", [])

    # Find the last assistant turn
    last_assistant = None
    for turn in reversed(conversations):
        if turn.get("from") in ("assistant", "gpt"):
            last_assistant = turn.get("value", "").strip()
            break
    if not last_assistant:
        return None, None

    # Build system content: inject function list from system field if present,
    # otherwise the system field already contains the ToolACE-format system prompt.
    system_content = system_raw if system_raw else ""

    role_map = {"human": "user", "gpt": "assistant", "assistant": "assistant", "user": "user"}
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})

    for turn in conversations:
        role = role_map.get(turn.get("from", ""), "user")
        content = turn.get("value", "").strip()
        if role == "assistant" and content == last_assistant:
            break  # stop before last assistant turn
        messages.append({"role": role, "content": content})

    return messages, last_assistant


# ---------------------------------------------------------------------------
# vLLM sampling — async, concurrent across examples
# ---------------------------------------------------------------------------

async def _sample_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    messages: list[dict],
    chosen_text: str,
    n: int,
    temperature: float,
    model_id: str = "/model",
) -> dict | None:
    """Return a preference pair dict or None if no rejected sample found."""
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 256,
        "temperature": temperature,
        "n": n,
        "stream": False,
    }
    async with semaphore:
        try:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
        except Exception as e:
            logger.debug("Sampling error: %s", e)
            return None

    samples = [c["message"]["content"] for c in data.get("choices", [])]
    chosen_names = _func_names(chosen_text)
    for s in samples:
        s_names = _func_names(s)
        if not s_names or s_names != chosen_names:
            return {
                "prompt":   messages,
                "chosen":   [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": s}],
            }
    return None  # model got everything right — skip


async def _generate_async(
    endpoint: str, examples: list[dict], n_samples: int, temperature: float,
    concurrency: int, min_calls: int = 1, model_id: str = "/model",
) -> tuple[list[dict], int]:
    url = f"{endpoint}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)

    valid = [(ex, *extract_prompt_and_answer(ex)) for ex in examples]
    valid = [(ex, msgs, ans) for ex, msgs, ans in valid if msgs is not None]
    if min_calls > 1:
        valid = [(ex, msgs, ans) for ex, msgs, ans in valid
                 if len(_parse_bracket_calls(ans)) >= min_calls]
        logger.info("After min_calls=%d filter: %d examples", min_calls, len(valid))
    n_skipped_invalid = len(examples) - len(valid)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            _sample_one(session, semaphore, url, msgs, ans, n_samples, temperature, model_id)
            for _, msgs, ans in valid
        ]
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if (i + 1) % 200 == 0:
                done = sum(1 for r in results if r is not None)
                logger.info("Progress: %d / %d  (pairs so far: %d)", i + 1, len(valid), done)

    records = [r for r in results if r is not None]
    n_skipped = n_skipped_invalid + (len(valid) - len(records))
    return records, n_skipped


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def _resolve_model_path(raw: str) -> str:
    """Resolve HuggingFace model ID to local cache path for Docker volume mount."""
    p = Path(raw)
    if p.is_absolute() or p.exists():
        resolved = str(p.resolve())
        if not Path(resolved).exists():
            logger.error("merged-path not found: %s", resolved)
            sys.exit(1)
        return resolved
    # HuggingFace model ID: find in local cache
    import huggingface_hub
    try:
        resolved = huggingface_hub.snapshot_download(raw, local_files_only=True)
        logger.info("Resolved HF model ID '%s' → %s", raw, resolved)
        return resolved
    except Exception as e:
        logger.error("Cannot resolve model '%s' locally: %s", raw, e)
        sys.exit(1)


def generate(args) -> None:
    merged_path = _resolve_model_path(args.merged_path)


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = f"http://localhost:{args.vllm_port}"

    proc, log_file, container_model = start_vllm(merged_path, args.vllm_port)
    try:
        if not wait_for_vllm(args.vllm_port):
            sys.exit(1)

        examples = load_toolace()
        if args.max_examples:
            examples = examples[: args.max_examples]

        logger.info("Generating with concurrency=%d, n_samples=%d", args.concurrency, args.n_samples)
        records, n_skipped = asyncio.run(
            _generate_async(endpoint, examples, args.n_samples, args.temperature,
                            args.concurrency, args.min_calls, container_model)
        )

        logger.info("Done. Pairs: %d  Skipped: %d  Total: %d", len(records), n_skipped, len(examples))

    finally:
        stop_vllm()
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.close()

    if not records:
        logger.error("No preference pairs generated — check vLLM logs")
        sys.exit(1)

    # Save dataset
    ds = Dataset.from_list(records)
    ds.save_to_disk(str(output_dir))

    stats = {
        "total_examples": len(examples),
        "pairs_generated": len(records),
        "skipped": n_skipped,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "merged_path": merged_path,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("Dataset saved to %s", output_dir)

    # MLflow logging
    mlflow.set_tracking_uri(args.mlflow_uri)
    exp = mlflow.get_experiment_by_name("orpo_dataset_gen")
    exp_id = exp.experiment_id if exp else mlflow.create_experiment("orpo_dataset_gen")
    with mlflow.start_run(run_name="gen_orpo_dataset", experiment_id=exp_id):
        mlflow.log_params({"merged_path": merged_path, "n_samples": args.n_samples,
                           "temperature": args.temperature, "max_examples": args.max_examples or "all"})
        mlflow.log_metrics({"pairs_generated": len(records), "skipped": n_skipped})
    logger.info("Stats logged to MLflow")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ORPO preference dataset from sft_qlora_r128")
    p.add_argument("--merged-path", required=True, help="Path to sft_qlora_r128/merged")
    p.add_argument("--output-dir", default="data/orpo_dataset")
    p.add_argument("--n-samples", type=int, default=6, help="vLLM samples per example")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--concurrency", type=int, default=16, help="Concurrent requests to vLLM")
    p.add_argument("--min-calls", type=int, default=1,
                   help="Min function calls in chosen (1=all, 2=parallel/multiple only)")
    p.add_argument("--max-examples", type=int, default=None, help="Cap on ToolACE examples (None = all)")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return p.parse_args()


if __name__ == "__main__":
    generate(parse_args())
