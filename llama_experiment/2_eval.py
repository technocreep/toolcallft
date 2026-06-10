"""
BFCL eval for a fine-tuned Llama-3.1-8B-Instruct trained on ToolACE.

The model outputs function calls in bracket notation:
    [func_name(arg1=val1, arg2=val2), func2(arg=val)]

A custom ToolACEHandler is registered that sends the ToolACE system prompt
(matching training format) and parses bracket notation via ast.parse.

Usage:
    python llama_experiment/2_eval.py --merged-path results/llama_experiment/sft_lora_r8/merged
    python llama_experiment/2_eval.py --merged-path results/llama_experiment/sft_lora_r32/merged \\
        --dtype fp8 --mlflow-run-name sft_lora_r32
"""

import argparse
import ast
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def log_bfcl_to_mlflow(metrics: dict) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
        else:
            mlflow.log_param(key, value)


# ---------------------------------------------------------------------------
# ToolACE bracket-notation parser
# ---------------------------------------------------------------------------

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
        func_name = _ast_name(elt.func)
        if not func_name:
            continue
        kwargs = {}
        for kw in elt.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        for i, arg in enumerate(elt.args):
            kwargs[f"positional_{i}"] = ast.literal_eval(arg)
        results.append({func_name: kwargs})

    return results


def _extract_first_bracket(text: str) -> str | None:
    depth = 0
    start = None
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


def _ast_name(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _calls_to_execute(calls: list[dict]) -> list[str]:
    out = []
    for call in calls:
        for name, kwargs in call.items():
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
            out.append(f"{name}({args_str})")
    return out


# ---------------------------------------------------------------------------
# Custom BFCL handler (Llama chat template)
# ---------------------------------------------------------------------------

def _build_toolace_handler():
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler

    TOOLACE_SYSTEM_PREFIX = (
        "You are an expert in composing functions. You are given a question and a set of possible functions. \n"
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. \n"
        "If none of the function can be used, point it out. If the given question lacks the parameters required "
        "by the function,\nalso point it out. You should only return the function call in tools call sections.\n"
        "Here is a list of functions in JSON format that you can invoke:\n"
    )
    TOOLACE_SYSTEM_SUFFIX = (
        ". \nShould you decide to return the function call(s). \n"
        "Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n\n"
        "NO other text MUST be included. \n"
    )

    def _pre_query_impl(self, test_entry: dict) -> dict:
        return {"message": [], "function": test_entry["function"]}
    _pre_query_impl.__override__ = True

    class ToolACEHandler(OSSHandler):
        def __init__(self, model_name, temperature, registry_name, is_fc_model, dtype="bfloat16", **kwargs):
            super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

        _pre_query_processing_prompting = _pre_query_impl

        def _format_prompt(self, messages, function):
            system_content = (
                TOOLACE_SYSTEM_PREFIX
                + json.dumps(function, ensure_ascii=False)
                + TOOLACE_SYSTEM_SUFFIX
            )
            # Llama 3 chat template
            formatted = "<|begin_of_text|>"
            formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{system_content}<|eot_id|>"
            for msg in messages:
                role = msg["role"]
                content = msg["content"].strip()
                formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return formatted
        _format_prompt.__override__ = True

        def decode_ast(self, result, language, has_tool_call_tag):
            calls = _parse_bracket_calls(result)
            if not calls:
                logger.debug("decode_ast: no bracket calls found in: %r", result[:200])
            return calls
        decode_ast.__override__ = True

        def decode_execute(self, result, has_tool_call_tag):
            calls = _parse_bracket_calls(result)
            return _calls_to_execute(calls)
        decode_execute.__override__ = True

    return ToolACEHandler


# ---------------------------------------------------------------------------
# BFCL evaluation
# ---------------------------------------------------------------------------

_CATEGORY_MAP = {
    "simple": "simple_python",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "irrelevance": "irrelevance",
}


def run_bfcl_toolace(endpoint: str, model_id: str, local_model_path: str) -> dict:
    from types import SimpleNamespace
    from urllib.parse import urlparse

    from bfcl_eval.constants.eval_config import RESULT_PATH
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from transformers import AutoTokenizer

    ToolACEHandler = _build_toolace_handler()

    if model_id not in MODEL_CONFIG_MAPPING:
        MODEL_CONFIG_MAPPING[model_id] = ModelConfig(
            model_name=model_id,
            display_name=model_id,
            url="",
            org=model_id.split("/")[0] if "/" in model_id else model_id,
            license="unknown",
            model_handler=ToolACEHandler,
            input_price=None,
            output_price=None,
            is_fc_model=False,
            underscore_to_dot=False,
        )

    _orig_spin_up = ToolACEHandler.spin_up_local_server

    def _patched_spin_up(self, *args, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        cfg_path = Path(local_model_path) / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text())
            self.max_context_length = cfg.get("max_position_embeddings", 131072)
        except Exception:
            self.max_context_length = 131072

    ToolACEHandler.spin_up_local_server = _patched_spin_up

    parsed = urlparse(endpoint)
    os.environ["LOCAL_SERVER_ENDPOINT"] = parsed.hostname or "localhost"
    os.environ["LOCAL_SERVER_PORT"] = str(parsed.port or 8000)

    categories = list(_CATEGORY_MAP.values())

    from bfcl_eval._llm_response_generation import main as generation_main
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

    logger.info("BFCL generate (ToolACE handler): model=%s categories=%s", model_id, categories)
    try:
        generation_main(gen_args)
    finally:
        ToolACEHandler.spin_up_local_server = _orig_spin_up

    import bfcl_eval.eval_checker.eval_runner as _runner
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    import tempfile

    captured: dict = {}
    _orig_record = _runner.record_result

    def _capture(table, model_name, test_category, accuracy, total_count):
        captured.setdefault(model_name, {})[test_category] = accuracy
        return _orig_record(table, model_name, test_category, accuracy, total_count)

    _runner.record_result = _capture
    try:
        with tempfile.TemporaryDirectory() as tmp:
            logger.info("BFCL evaluate: model=%s", model_id)
            evaluation_main(model=[model_id], test_categories=categories,
                            result_dir=None, score_dir=tmp, partial_eval=False)
    finally:
        _runner.record_result = _orig_record

    slug = model_id.replace("/", "_")
    model_data = captured.get(model_id) or captured.get(slug) or {}

    metrics: dict = {}
    scores: list[float] = []
    for metric_key, cat_name in _CATEGORY_MAP.items():
        acc = model_data.get(cat_name)
        if acc is not None:
            metrics[f"bfcl_python_{metric_key}"] = float(acc)
            scores.append(float(acc))
        else:
            logger.warning("No score for category '%s'", cat_name)
    metrics["bfcl_python_overall"] = sum(scores) / len(scores) if scores else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Docker vLLM lifecycle
# ---------------------------------------------------------------------------

def evaluate(merged_path: str, port: int, dtype: str, mlflow_run_name: str | None, mlflow_uri: str) -> None:
    import urllib.request
    import json as _json

    merged_path = str(Path(merged_path).resolve())
    if not Path(merged_path).exists():
        logger.error("Merged model path does not exist: %s", merged_path)
        sys.exit(1)

    vllm_dtype = "bfloat16" if dtype in ("bf16", "bfloat16") else dtype

    container_model_path = "/model"
    container_name = f"vllm-llama-eval-{port}"
    log_path = Path(f"/tmp/vllm-llama-eval-{port}.log")
    log_file = log_path.open("w")

    cmd = [
        "sudo", "docker", "run", "--rm",
        "--name", container_name,
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{port}:{port}",
        "-v", f"{merged_path}:{container_model_path}",
        "vllm/vllm-openai:latest",
        "--model", container_model_path,
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--dtype", vllm_dtype,
        "--trust-remote-code",
        "--max-model-len", "8192",
    ]
    if dtype == "fp8":
        cmd += ["--quantization", "fp8"]

    logger.info("Starting vLLM container (dtype=%s, logs → %s)", dtype, log_path)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

    try:
        url = f"http://localhost:{port}/v1/models"
        time.sleep(10)
        deadline = time.time() + 300
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                logger.error("vLLM container exited prematurely (code %d)", proc.returncode)
                logger.error("Check logs: %s", log_path)
                return
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = _json.loads(resp.read())
                    served = [m["id"] for m in data.get("data", [])]
                    if any(container_model_path in s or s in container_model_path for s in served):
                        logger.info("vLLM server ready on port %d", port)
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(5)

        if not ready:
            logger.error("vLLM server did not become ready within 300s")
            return

        metrics = run_bfcl_toolace(
            endpoint=f"http://localhost:{port}",
            model_id=container_model_path,
            local_model_path=merged_path,
        )
        logger.info("BFCL (ToolACE handler) overall=%.4f", metrics.get("bfcl_python_overall", 0.0))
        logger.info("Per-category: %s", {k: f"{v:.3f}" for k, v in metrics.items()})

        mlflow.set_tracking_uri(mlflow_uri)
        run_name = mlflow_run_name or (Path(merged_path).parent.name + "_llama_toolace")
        exp = mlflow.get_experiment_by_name("fine_tuning")
        experiment_id = exp.experiment_id if exp else mlflow.create_experiment("fine_tuning")

        with mlflow.start_run(run_name=f"{run_name}_toolace_eval", experiment_id=experiment_id):
            mlflow.log_param("merged_path", merged_path)
            mlflow.log_param("eval_handler", "ToolACEHandler_Llama")
            mlflow.log_param("dtype", dtype)
            log_bfcl_to_mlflow(metrics)
        logger.info("Logged to MLflow run '%s_toolace_eval'", run_name)

        out = Path(merged_path).parent / "bfcl_metrics_toolace.json"
        out.write_text(json.dumps(metrics, indent=2))
        logger.info("Saved metrics to %s", out)

    finally:
        subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_file.close()
        logger.info("vLLM container stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BFCL eval for Llama fine-tuned on ToolACE (bracket-notation handler)"
    )
    parser.add_argument("--merged-path", required=True, help="Path to merged model directory")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp8"],
        help="vLLM serving dtype: bf16 (default) or fp8 (quantized, faster on H100)",
    )
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        merged_path=args.merged_path,
        port=args.vllm_port,
        dtype=args.dtype,
        mlflow_run_name=args.mlflow_run_name,
        mlflow_uri=args.mlflow_uri,
    )
