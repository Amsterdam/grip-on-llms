"""Calculate costs for different models"""
import json
from collections import defaultdict
from datetime import datetime

import tiktoken

# Load the file
with open("leaderboard_final", "r", encoding="utf-8") as f:
    data = json.load(f)

# Included benchmarks
included_benchmarks = {
    "AmsterdamSimplification-detailed",
    "INT_Duidelijke_Taal-detailed",
    "CNNDailyMail",
    "XSum",
}

# API pricing table ($ per 1k tokens)
api_model_pricing = {
    "gpt-4o": {"input": 0.005, "output": 0.02},
    "gpt-4o-mini": {"input": 0.0006, "output": 0.0024},
}

# Current hourly GPU rates on Azure
gpu_hourly_rates = {
    "Tesla T4": 0.66,
    "H100": 9.08,
    "Tesla V100-PCIE-16GB": 3.82,
}


def parse_duration(start, end):
    """Parse duration in seconds per benchmark run."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start_time = datetime.strptime(start, fmt)
    end_time = datetime.strptime(end, fmt)
    return (end_time - start_time).total_seconds()


def count_tokens(model_name, text):
    """Count tokens using tiktoken for a given model and text."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
    except Exception:
        return 0


results = defaultdict(lambda: defaultdict(dict))

# Process entries in leaderboard dataframe
for entry in data:
    try:
        metadata = entry.get("metadata")
        model = metadata.get("llm", {}).get("model_name")
        benchmark = metadata.get("benchmark", {}).get("name")
        if benchmark not in included_benchmarks:
            continue

        n_tokens = metadata.get("n_tokens")
        run_output = entry.get("benchmark_results", {}).get("run_output", [])
        n_samples = metadata.get("n_samples", 1)
        run = metadata.get("run")
        device = run.get("system", {}).get("device_info", {}).get("gpu", {}).get("device_name")
        start_time, end_time = run.get("timestamp_bench_start"), run.get("timestamp_bench_end")

        if model in api_model_pricing:
            n_input = n_tokens.get("n_input_tokens") if isinstance(n_tokens, dict) else None
            n_output = n_tokens.get("n_output_tokens") if isinstance(n_tokens, dict) else None

            if n_input is None or n_output is None:
                inputs = [r.get("prompt") or r.get("source") for r in run_output]
                outputs = [r.get("response") for r in run_output]
                n_input = sum(count_tokens(model, p) for p in inputs if isinstance(p, str))
                n_output = sum(count_tokens(model, o) for o in outputs if isinstance(o, str))

            res = results[benchmark].setdefault(model, {"type": "API", "input": 0, "output": 0})
            res["input"] += n_input
            res["output"] += n_output

        else:
            duration = parse_duration(start_time, end_time)
            gpu_rate = gpu_hourly_rates.get(device)
            if duration and gpu_rate:
                cost = duration * gpu_rate / 3600
                res = results[benchmark].setdefault(model, {"type": "Open Source", "costs": []})
                res["costs"].extend([cost / n_samples] * n_samples)
            else:
                continue

    except Exception as e:
        print(f"Skipping due to error: {e}")
        continue

# Combine results from API and open source models
cost_results = {}
for benchmark, models in results.items():
    cost_results[benchmark] = {}
    for model, stats in models.items():
        if stats["type"] == "API":
            pricing = api_model_pricing[model]
            in_tokens, out_tokens = stats["input"], stats["output"]
            cost_per_prompt = in_tokens * (pricing["input"] / 1000) + out_tokens * (
                pricing["output"] / 1000
            )
            cost_results[benchmark][model] = {
                "type": "API",
                "avg_input_tokens": round(in_tokens, 2),
                "avg_output_tokens": round(out_tokens, 2),
                "cost_per_prompt": round(cost_per_prompt, 6),
            }
        else:
            costs = stats["costs"]
            cost_results[benchmark][model] = {
                "type": "Open Source",
                "avg_cost_per_prompt": round(sum(costs) / len(costs), 6),
            }

# Average cost per model across included benchmarks
model_costs = defaultdict(lambda: {"total_cost": 0.0, "total_samples": 0})

for models in cost_results.values():
    for model, stats in models.items():
        cost = stats.get("cost_per_prompt") or stats.get("avg_cost_per_prompt")
        model_costs[model]["total_cost"] += cost
        model_costs[model]["total_samples"] += 1

# Compute average cost per model
avg_costs_per_model = {}
for model, values in model_costs.items():
    total = values["total_cost"]
    n_samples = values["total_samples"]
    avg_costs_per_model[model] = round(total / n_samples, 6)

costs_sorted = sorted(avg_costs_per_model.items(), key=lambda x: x[1])
print(costs_sorted)
