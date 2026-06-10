"""
Latency benchmark for a fine-tuned Llama model served via vLLM.

Requests are formatted in ToolACE style (function list in system prompt + user query),
matching the training distribution. The model responds in bracket notation:
  [func_name(param=value, ...)]

Metrics:
  TTFT      — time to first token (mean, p50, p95, p99)
  E2E       — end-to-end latency per request
  RPS       — requests per second
  tok/s     — output tokens per second
  tool_call% — fraction of responses containing a bracket-notation call

Usage:
    python llama_experiment/3_bench.py --merged-path results/llama_experiment/sft_lora_r128/merged
    python llama_experiment/3_bench.py --merged-path results/llama_experiment/sft_lora_r128/merged --quantization fp8
    python llama_experiment/3_bench.py --host localhost --port 8000  # external server
"""

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------------

CONTAINER_NAME = "vllm-llama-bench"
CONTAINER_MODEL = "/model"


def start_vllm(merged_path: str, port: int, quantization: str | None) -> tuple:
    cmd = [
        "sudo", "docker", "run", "--rm",
        "--name", CONTAINER_NAME,
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{port}:{port}",
        "-v", f"{merged_path}:{CONTAINER_MODEL}",
        "vllm/vllm-openai:latest",
        "--model", CONTAINER_MODEL,
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--disable-log-stats",
        "--max-model-len", "4096",
    ]
    if quantization and quantization.lower() not in ("none", "bf16", "bfloat16", ""):
        cmd += ["--quantization", quantization]

    log_path = Path(f"/tmp/vllm-bench-{port}.log")
    log_file = log_path.open("w")
    print(f"Starting vLLM container (quantization={quantization or 'none'})...")
    print(f"Logs → {log_path}")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    return proc, log_file


def wait_for_vllm(port: int, timeout: int = 300) -> str | None:
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    print("Waiting for vLLM", end="", flush=True)
    time.sleep(20)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("data"):
                    model_id = data["data"][0]["id"]
                    print(f" ready! (model={model_id})")
                    return model_id
        except Exception:
            print(".", end="", flush=True)
            time.sleep(5)
    print(" TIMEOUT")
    return None


def stop_vllm():
    subprocess.run(
        ["sudo", "docker", "stop", CONTAINER_NAME],
        check=False, capture_output=True,
    )
    print("vLLM container stopped.")


# ---------------------------------------------------------------------------
# ToolACE-style test payloads
# Functions and queries match the ToolACE training distribution.
# System prompt format is identical to training: JSON function list +
# bracket-notation instruction.
# ---------------------------------------------------------------------------

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

# Function definitions in ToolACE JSON schema format
FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a specified city",
        "parameters": {
            "type": "dict",
            "properties": {
                "city":  {"type": "string", "description": "The city name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                          "description": "Temperature units"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": "Search the web for information on a given topic",
        "parameters": {
            "type": "dict",
            "properties": {
                "query":       {"type": "string", "description": "Search query string"},
                "num_results": {"type": "integer", "description": "Number of results to return"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result",
        "parameters": {
            "type": "dict",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol",
        "parameters": {
            "type": "dict",
            "properties": {
                "ticker":   {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                "currency": {"type": "string", "description": "Currency for the price, default USD"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another",
        "parameters": {
            "type": "dict",
            "properties": {
                "amount":        {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code, e.g. USD"},
                "to_currency":   {"type": "string", "description": "Target currency code, e.g. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]

# Queries representative of ToolACE test cases
QUERIES = [
    "What's the weather like in Berlin right now?",
    "Search for the latest news about artificial intelligence.",
    "Calculate 15% tip on a $47.50 restaurant bill.",
    "What is the current temperature in Tokyo in celsius?",
    "Find information about best practices in Python async programming.",
    "What's the weather in Paris and London?",
    "Calculate the compound interest: principal 10000, rate 5%, years 3.",
    "Search for recent papers on large language model fine-tuning.",
    "Is it going to rain in Amsterdam today?",
    "What is 2 to the power of 32?",
    "Get me the stock price of Apple.",
    "Convert 500 USD to EUR.",
    "What's the current weather in New York in fahrenheit?",
    "Search for 'transformer architecture attention mechanism' and return 5 results.",
    "Calculate sqrt(144) + 15 * 3.",
    "Get the stock price for TSLA and NVDA.",
]


def _build_system_prompt() -> str:
    return TOOLACE_SYSTEM_PREFIX + json.dumps(FUNCTIONS, ensure_ascii=False) + TOOLACE_SYSTEM_SUFFIX


def make_payload(query: str, model_id: str) -> dict:
    return {
        "model":       model_id,
        "messages": [
            {"role": "system",  "content": _build_system_prompt()},
            {"role": "user",    "content": query},
        ],
        "max_tokens":  256,
        "temperature": 0.0,
        "stream":      True,
    }


def _has_tool_call(text: str) -> bool:
    """Check if response contains bracket-notation tool call."""
    return bool(re.search(r"\[\w+\(", text))


# ---------------------------------------------------------------------------
# Async streaming request
# ---------------------------------------------------------------------------

async def send_request(
    session:   aiohttp.ClientSession,
    url:       str,
    payload:   dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        t_start   = time.perf_counter()
        ttft      = None
        tokens    = 0
        full_text = []

        try:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {"error": f"HTTP {resp.status}: {body[:200]}"}

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta   = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content") or ""

                    if ttft is None and content:
                        ttft = time.perf_counter() - t_start

                    if content:
                        tokens += len(content.split())
                        full_text.append(content)

        except asyncio.TimeoutError:
            return {"error": "timeout"}
        except Exception as e:
            return {"error": str(e)}

        e2e      = time.perf_counter() - t_start
        response = "".join(full_text)
        return {
            "ttft_s":    ttft if ttft is not None else e2e,
            "e2e_s":     e2e,
            "tokens":    tokens,
            "has_call":  _has_tool_call(response),
            "error":     None,
        }


# ---------------------------------------------------------------------------
# Benchmark one concurrency level
# ---------------------------------------------------------------------------

async def run_benchmark(url: str, model_id: str, concurrency: int, num_requests: int) -> dict:
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 8)

    payloads = [
        make_payload(QUERIES[i % len(QUERIES)], model_id)
        for i in range(num_requests)
    ]

    print(f"\n  concurrency={concurrency}, requests={num_requests}", flush=True)

    t_wall = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector) as session:
        results = await asyncio.gather(
            *[send_request(session, url, p, semaphore) for p in payloads]
        )
    t_wall = time.perf_counter() - t_wall

    ok     = [r for r in results if r.get("error") is None]
    errors = [r for r in results if r.get("error") is not None]

    if not ok:
        print(f"  ERROR: all {len(errors)} requests failed")
        for e in errors[:3]:
            print(f"    {e['error']}")
        return {"concurrency": concurrency, "ok": 0, "errors": len(errors)}

    def pct(lst, p):
        s = sorted(lst)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    ttfts      = [r["ttft_s"]   for r in ok]
    e2es       = [r["e2e_s"]    for r in ok]
    tokens     = sum(r["tokens"] for r in ok)
    tool_calls = sum(1 for r in ok if r["has_call"])

    stats = {
        "concurrency":     concurrency,
        "ok":              len(ok),
        "errors":          len(errors),
        "wall_time_s":     round(t_wall, 3),
        "throughput_rps":  round(len(ok) / t_wall, 2),
        "tool_call_pct":   round(tool_calls / len(ok) * 100, 1),
        "ttft_mean_ms":    round(statistics.mean(ttfts) * 1000, 1),
        "ttft_p50_ms":     round(pct(ttfts, 50) * 1000, 1),
        "ttft_p95_ms":     round(pct(ttfts, 95) * 1000, 1),
        "ttft_p99_ms":     round(pct(ttfts, 99) * 1000, 1),
        "e2e_mean_ms":     round(statistics.mean(e2es) * 1000, 1),
        "e2e_p50_ms":      round(pct(e2es, 50) * 1000, 1),
        "e2e_p95_ms":      round(pct(e2es, 95) * 1000, 1),
        "tokens_per_sec":  round(tokens / t_wall, 1),
    }

    print(f"  OK={len(ok)} tool_call%={stats['tool_call_pct']} | "
          f"TTFT mean={stats['ttft_mean_ms']}ms p95={stats['ttft_p95_ms']}ms | "
          f"E2E mean={stats['e2e_mean_ms']}ms | RPS={stats['throughput_rps']}")
    return stats


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

async def warmup(url: str, model_id: str, n: int = 8):
    print(f"Warming up ({n} requests)...")
    sem = asyncio.Semaphore(n)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            send_request(session, url, make_payload(QUERIES[i % len(QUERIES)], model_id), sem)
            for i in range(n)
        ])
    print("Warmup done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main(args):
    base_url = f"http://{args.host}:{args.port}"
    chat_url = f"{base_url}/v1/chat/completions"

    print(f"Connecting to {base_url}...")
    async with aiohttp.ClientSession() as s:
        async with s.get(
            f"{base_url}/v1/models",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            data     = await r.json()
            model_id = data["data"][0]["id"]
    print(f"Model: {model_id}")

    await warmup(chat_url, model_id)

    all_stats = []
    for conc in args.concurrency:
        stats = await run_benchmark(
            url=chat_url,
            model_id=model_id,
            concurrency=conc,
            num_requests=args.num_requests,
        )
        all_stats.append(stats)

    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Conc':>6} {'OK':>5} {'call%':>6} {'RPS':>6} "
          f"{'TTFT mean':>10} {'TTFT p50':>9} {'TTFT p95':>9} {'TTFT p99':>9} "
          f"{'E2E mean':>9} {'E2E p95':>8} {'tok/s':>7}")
    print("-" * 100)
    for s in all_stats:
        if s.get("ok", 0) == 0:
            print(f"{s['concurrency']:>6}  ALL FAILED")
            continue
        print(
            f"{s['concurrency']:>6} "
            f"{s['ok']:>5} "
            f"{s['tool_call_pct']:>5.0f}% "
            f"{s['throughput_rps']:>6.1f} "
            f"{s['ttft_mean_ms']:>9.0f}ms "
            f"{s['ttft_p50_ms']:>8.0f}ms "
            f"{s['ttft_p95_ms']:>8.0f}ms "
            f"{s['ttft_p99_ms']:>8.0f}ms "
            f"{s['e2e_mean_ms']:>8.0f}ms "
            f"{s['e2e_p95_ms']:>7.0f}ms "
            f"{s['tokens_per_sec']:>6.0f}"
        )
    print("=" * 100)

    for target in (16, 32, 64):
        s = next((x for x in all_stats if x.get("concurrency") == target), None)
        if s and s.get("ok", 0) > 0:
            print(f"\n=== Production target: {target} concurrent requests ===")
            print(f"  Tool call rate: {s['tool_call_pct']:.0f}%")
            print(f"  TTFT mean : {s['ttft_mean_ms']:.0f} ms")
            print(f"  TTFT p95  : {s['ttft_p95_ms']:.0f} ms")
            print(f"  TTFT p99  : {s['ttft_p99_ms']:.0f} ms")
            print(f"  E2E  mean : {s['e2e_mean_ms']:.0f} ms")
            print(f"  E2E  p95  : {s['e2e_p95_ms']:.0f} ms")
            print(f"  Throughput: {s['throughput_rps']:.1f} req/s")
            print(f"  Tokens/s  : {s['tokens_per_sec']:.0f}")

    if args.output:
        out = Path(args.output)
    elif args.merged_path:
        out = Path(args.merged_path).parent / "bench_results.json"
    else:
        out = Path("logs/llama_bench_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_stats, indent=2))
    print(f"\nSaved → {out}")

    return all_stats


def main():
    args = parse_args()

    if args.merged_path:
        merged_path = str(Path(args.merged_path).resolve())
        proc, log_file = start_vllm(merged_path, args.port, args.quantization)
        model_id = wait_for_vllm(args.port)
        if not model_id:
            print("Failed to start vLLM")
            stop_vllm()
            log_file.close()
            return
        try:
            asyncio.run(async_main(args))
        finally:
            stop_vllm()
            log_file.close()
    else:
        asyncio.run(async_main(args))


def parse_args():
    p = argparse.ArgumentParser(description="ToolACE-format latency benchmark for Llama fine-tuned models")
    p.add_argument(
        "--merged-path", default=None,
        help="Path to merged model. If set, script starts vLLM itself.",
    )
    p.add_argument(
        "--quantization", default=None,
        help="vLLM quantization: fp8, awq, none/omit for bf16",
    )
    p.add_argument("--host",         default="localhost")
    p.add_argument("--port",         type=int, default=8000)
    p.add_argument(
        "--concurrency", type=int, nargs="+", default=[8, 16, 32, 64],
        help="Concurrency levels to test",
    )
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument(
        "--output", default=None,
        help="Path for results JSON. Default: <run_dir>/bench_results.json",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
