"""
Script 1: Baseline Evaluation

For each candidate model:
  1. Launch as OpenAI-compatible vLLM server
  2. Run BFCL Python subset evaluation
  3. Log metrics to MLflow under experiment 'baseline_eval'
  4. Save per-model results to results/baseline_eval/{model_name}/metrics.json
  5. Shut down server, load next model

Usage:
    python scripts/1_baseline_eval.py [--models all] [--vllm-port 8000]
                                      [--mlflow-uri ./mlruns]
                                      [--results-dir ./results/baseline_eval]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml

from dotenv import load_dotenv

load_dotenv()

# Allow running from project root or scripts/ directory
sys.path.insert(0, str(Path(__file__).parent))
from bfcl_utils import log_bfcl_to_mlflow, run_bfcl_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
BFCL_EVAL_VERSION = "2025.12.17"


# ---------------------------------------------------------------------------
# vLLM server lifecycle
# ---------------------------------------------------------------------------

def start_vllm_server(
    model_id: str,
    port: int,
    gpu_memory_util: float = 0.90,
    max_model_len: int | None = None,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server in a Docker container."""
    hf_cache = Path.home() / ".cache" / "huggingface"
    container_name = f"vllm-{model_id.replace('/', '-')}-{port}"
    cmd = [
        "sudo", "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{port}:{port}",
        "-v", f"{hf_cache}:/root/.cache/huggingface",
        "-e", f"HF_TOKEN={os.environ['HF_TOKEN']}",
        "vllm/vllm-openai:latest",
        "--model", model_id,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    log_path = Path(f"/tmp/vllm-{port}.log")
    log_file = log_path.open("w")
    logger.info("Starting vLLM container: %s  (logs → %s)", " ".join(cmd), log_path)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    proc._log_file = log_file  # keep reference so it isn't GC'd
    return proc


def wait_for_server(port: int, model_id: str, proc: subprocess.Popen, timeout: int = 300) -> bool:
    """Poll /v1/models until the correct model is loaded or timeout is reached.

    Returns False immediately if the container process exits before becoming ready.
    """
    import json as _json
    import urllib.request
    url = f"http://localhost:{port}/v1/models"
    time.sleep(10)  # let old container release the port before first poll
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            logger.error("vLLM container exited prematurely (exit code %d)", proc.returncode)
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = _json.loads(resp.read())
                served = [m["id"] for m in data.get("data", [])]
                if any(model_id in s or s in model_id for s in served):
                    logger.info("vLLM server ready on port %d (model=%s)", port, served)
                    return True
        except Exception:
            pass
        time.sleep(5)
    logger.error("vLLM server did not become ready within %ds", timeout)
    return False


def stop_vllm_server(proc: subprocess.Popen) -> None:
    """Stop the vLLM Docker container and wait for the process to exit."""
    # Find the container name from the original command args
    try:
        args = proc.args  # list passed to Popen
        name_idx = args.index("--name") + 1
        container_name = args[name_idx]
        subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)
    except (ValueError, IndexError, TypeError):
        pass
    if proc.poll() is None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    log_file = getattr(proc, "_log_file", None)
    if log_file:
        log_file.close()
    logger.info("vLLM container stopped (pid=%d)", proc.pid)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_id: str,
    port: int,
    results_dir: Path,
    mlflow_experiment: str,
    gpu_memory_util: float = 0.90,
    max_model_len: int | None = None,
) -> dict:
    """Run baseline evaluation for a single model. Returns metrics dict."""
    model_slug = model_id.replace("/", "--")
    model_results_dir = results_dir / model_slug
    model_results_dir.mkdir(parents=True, exist_ok=True)

    endpoint = f"http://localhost:{port}"
    proc = start_vllm_server(model_id, port, gpu_memory_util=gpu_memory_util, max_model_len=max_model_len)

    try:
        ready = wait_for_server(port, model_id, proc)
        if not ready:
            raise RuntimeError(f"vLLM server for {model_id} failed to start")

        metrics = run_bfcl_eval(endpoint=endpoint, model_id=model_id)

        metrics_out = {
            "model_id": model_id,
            **metrics,
            "eval_timestamp": datetime.now(timezone.utc).isoformat(),
            "bfcl_eval_version": BFCL_EVAL_VERSION,
        }

        out_path = model_results_dir / "metrics.json"
        out_path.write_text(json.dumps(metrics_out, indent=2))
        logger.info("Saved metrics to %s", out_path)

        with mlflow.start_run(run_name=model_slug, experiment_id=_get_experiment_id(mlflow_experiment)):
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("bfcl_eval_version", BFCL_EVAL_VERSION)
            log_bfcl_to_mlflow(metrics)
            try:
                mlflow.log_artifact(str(out_path))
            except Exception as e:
                logger.warning("mlflow.log_artifact failed (artifact store misconfigured?): %s", e)

        return metrics_out

    finally:
        stop_vllm_server(proc)


def _get_experiment_id(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        return mlflow.create_experiment(name)
    return exp.experiment_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_model_configs(models_arg: str) -> list[dict]:
    """Return list of per-model config dicts from models.yaml, filtered by models_arg."""
    config_path = CONFIGS_DIR / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    vllm_defaults = config.get("vllm_defaults", {})
    all_models: list[dict] = config["candidate_models"]

    # Merge per-model entries with vllm_defaults so each dict is self-contained
    for m in all_models:
        for key, val in vllm_defaults.items():
            m.setdefault(key, val)

    if models_arg.strip().lower() == "all":
        return all_models

    all_ids = [m["id"] for m in all_models]
    requested = [m.strip() for m in models_arg.split(",")]
    resolved = []
    for req in requested:
        if req in all_ids:
            resolved.append(next(m for m in all_models if m["id"] == req))
        else:
            matches = [m for m in all_models if m["id"].split("/")[-1] == req]
            if matches:
                resolved.extend(matches)
            else:
                logger.warning("Model '%s' not found in models.yaml — skipping", req)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline BFCL evaluation for all candidate models")
    parser.add_argument("--models", default="all", help="Comma-separated model IDs or 'all'")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--mlflow-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--results-dir", default="./results/baseline_eval")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if results already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.now(timezone.utc)
    logger.info("=== Baseline Evaluation START %s ===", start_time.isoformat())

    mlflow.set_tracking_uri(args.mlflow_uri)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_configs = load_model_configs(args.models)
    logger.info("Models to evaluate: %s", [m["id"] for m in model_configs])

    all_results = []
    for mcfg in model_configs:
        model_id = mcfg["id"]
        model_slug = model_id.replace("/", "--")
        existing = results_dir / model_slug / "metrics.json"
        if existing.exists() and not args.force:
            cached = json.loads(existing.read_text())
            logger.info("--- Skipping %s (results exist, use --force to re-run) ---", model_id)
            all_results.append(cached)
            continue
        logger.info("--- Evaluating: %s ---", model_id)
        try:
            result = evaluate_model(
                model_id=model_id,
                port=args.vllm_port,
                results_dir=results_dir,
                mlflow_experiment="baseline_eval",
                gpu_memory_util=mcfg.get("gpu_memory_utilization", 0.90),
                max_model_len=mcfg.get("max_model_len"),
            )
            all_results.append(result)
            logger.info(
                "%s | overall=%.4f", model_id, result.get("bfcl_python_overall", 0.0)
            )
        except Exception:
            logger.exception("Failed to evaluate %s", model_id)

    end_time = datetime.now(timezone.utc)
    logger.info("=== Baseline Evaluation END %s ===", end_time.isoformat())
    logger.info("Duration: %s", end_time - start_time)
    logger.info("Results summary:")
    for r in all_results:
        logger.info("  %-50s overall=%.4f", r["model_id"], r.get("bfcl_python_overall", 0.0))


if __name__ == "__main__":
    main()
