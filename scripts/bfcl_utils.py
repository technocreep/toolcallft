"""
Shared helpers for BFCL evaluation.

run_bfcl_eval        — runs the Python subset against an OpenAI-compatible endpoint
log_bfcl_to_mlflow   — logs the returned metrics dict to the active MLflow run
"""

import logging
import os
import tempfile
from types import SimpleNamespace
from typing import Optional
from urllib.parse import urlparse

import mlflow

logger = logging.getLogger(__name__)

# Internal metric key → bfcl CLI test-category name
_CATEGORY_MAP: dict[str, str] = {
    "simple": "simple_python",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "irrelevance": "irrelevance",
}


def _pick_handler(model_id: str):
    """Return the best-fit bfcl_eval handler class for a model we don't recognise.

    LlamaHandler uses the standard BFCL system prompt with Llama chat tokens
    and expects pythonic [func(args)] output — correct for all Llama-based
    tool-calling models (including ToolACE).  QwenHandler is the fallback for
    everything else.
    """
    model_lower = model_id.lower()
    if any(k in model_lower for k in ("llama", "toolace")):
        from bfcl_eval.model_handler.local_inference.llama import LlamaHandler
        return LlamaHandler
    from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
    return QwenHandler


def _ensure_model_registered(model_id: str) -> None:
    """Register an unsupported model in bfcl_eval's MODEL_CONFIG_MAPPING."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    if model_id not in MODEL_CONFIG_MAPPING:
        handler = _pick_handler(model_id)
        MODEL_CONFIG_MAPPING[model_id] = ModelConfig(
            model_name=model_id,
            display_name=model_id,
            url="",
            org=model_id.split("/")[0] if "/" in model_id else model_id,
            license="unknown",
            model_handler=handler,
            input_price=None,
            output_price=None,
            is_fc_model=False,
            underscore_to_dot=False,
        )
        logger.info(
            "Registered '%s' in bfcl_eval model registry (handler=%s)",
            model_id,
            handler.__name__,
        )


def run_bfcl_eval(
    endpoint: str,
    model_id: str,
    max_tasks: Optional[int] = 150,
    seed: int = 42,
) -> dict:
    """
    Run BFCL Python-subset evaluation against a vLLM OpenAI-compatible server.

    Calls bfcl_eval Python API in-process (generate then evaluate).

    Returns:
        Flat dict: {"bfcl_python_overall": float, "bfcl_python_simple": float, ...}
    """
    _ensure_model_registered(model_id)

    parsed = urlparse(endpoint)
    os.environ["LOCAL_SERVER_ENDPOINT"] = parsed.hostname or "localhost"
    os.environ["LOCAL_SERVER_PORT"] = str(parsed.port or 8000)

    categories = list(_CATEGORY_MAP.values())

    # --- Generate ---
    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.constants.eval_config import RESULT_PATH

    gen_args = SimpleNamespace(
        model=[model_id],
        test_category=categories,
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_gpus=1,
        num_threads=None,
        gpu_memory_utilization=0.9,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=None,
        result_dir=RESULT_PATH,
        allow_overwrite=True,
        run_ids=False,
    )
    logger.info("BFCL generate: model=%s categories=%s endpoint=%s", model_id, categories, endpoint)
    generation_main(gen_args)

    # --- Evaluate (monkey-patch record_result to capture per-category scores) ---
    # eval_runner does `from eval_runner_helper import *`, so we must patch the
    # name inside the eval_runner module itself, not the helper module.
    import bfcl_eval.eval_checker.eval_runner as _runner
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    captured: dict[str, dict[str, float]] = {}
    _original_record = _runner.record_result

    def _capture_record(table, model_name, test_category, accuracy, total_count):
        captured.setdefault(model_name, {})[test_category] = accuracy
        return _original_record(table, model_name, test_category, accuracy, total_count)

    _runner.record_result = _capture_record
    try:
        # Use a throwaway score_dir so generate_leaderboard_csv doesn't scan
        # results from other models (which may not be registered yet).
        with tempfile.TemporaryDirectory() as tmp_score_dir:
            logger.info("BFCL evaluate: model=%s categories=%s", model_id, categories)
            evaluation_main(
                model=[model_id],
                test_categories=categories,
                result_dir=None,
                score_dir=tmp_score_dir,
                partial_eval=False,
            )
    finally:
        _runner.record_result = _original_record

    return _build_metrics(model_id, captured)


def _build_metrics(model_id: str, captured: dict) -> dict:
    # eval_runner stores results under directory name (slashes → underscores)
    slug = model_id.replace("/", "_")
    model_data = captured.get(model_id) or captured.get(slug) or {}

    metrics: dict = {}
    scores: list[float] = []
    for metric_key, category_name in _CATEGORY_MAP.items():
        acc = model_data.get(category_name)
        if acc is not None:
            metrics[f"bfcl_python_{metric_key}"] = float(acc)
            scores.append(float(acc))
        else:
            logger.warning("No score captured for category '%s'", category_name)

    metrics["bfcl_python_overall"] = sum(scores) / len(scores) if scores else 0.0
    return metrics


def log_bfcl_to_mlflow(metrics: dict) -> None:
    """Log a metrics dict produced by run_bfcl_eval to the active MLflow run."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
        else:
            mlflow.log_param(key, value)
