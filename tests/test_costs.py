"""Basic tests to calculate costs for different models"""
import json
from collections import defaultdict
from datetime import datetime

# Load the file
file_path = "leaderboard"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# API pricing table ($ per 1k tokens)
api_model_pricing = {
    "gpt-4o": {"input": 0.005, "output": 0.02},
    "gpt-4o-mini": {"input": 0.0006, "output": 0.0024},
}

# Current hourly GPU rates on Azure
gpu_hourly_rates = {
    "Tesla T4": 0.66,
    "A100": 9.55,
}

results = defaultdict(lambda: defaultdict(dict))


def parse_duration(start, end):
    """Parse duration in seconds per benchmark run."""
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start_dt = datetime.strptime(start, fmt)
    end_dt = datetime.strptime(end, fmt)
    return (end_dt - start_dt).total_seconds()


# Process entries in leaderboard dataframe
for entry in data:
    try:
        model = entry.get("metadata", {}).get("llm", {}).get("model_name")
        benchmark = entry.get("metadata", {}).get("benchmark", {}).get("name")
        n_tokens = entry.get("metadata", {}).get("n_tokens")
        n_samples = entry.get("metadata", {}).get("n_samples", 1)
        run = entry.get("metadata", {}).get("run")
        system = run.get("system", {}) if run else {}
        gpu_info = system.get("device_info", {}).get("gpu", {})
        device = gpu_info.get("device_name")
        start_time = run.get("timestamp_bench_start") if run else None
        end_time = run.get("timestamp_bench_end") if run else None

        if model in api_model_pricing:
            if (
                n_tokens
                and n_tokens.get("n_input_tokens") is not None
                and n_tokens.get("n_output_tokens") is not None
            ):
                tokens = results[benchmark].setdefault(
                    model, {"type": "API", "input": [], "output": []}
                )
                tokens["input"] = n_tokens["n_input_tokens"]
                tokens["output"] = n_tokens["n_output_tokens"]
            else:
                continue
        else:
            duration = parse_duration(start_time, end_time)
            gpu_rate = gpu_hourly_rates.get(device)
            if duration is not None and gpu_rate is not None:
                cost = duration * (gpu_rate / 3600)
                per_prompt_cost = cost / n_samples
                durations = results[benchmark].setdefault(
                    model, {"type": "Open Source", "costs": []}
                )
                durations["costs"].extend([per_prompt_cost] * n_samples)
            else:
                continue
    except Exception as e:
        print(f"Skipping due to error: {e}")
        continue

# Combine results from API and open source models
final_results = {}
for benchmark, models in results.items():
    final_results[benchmark] = {}
    for model, stats in models.items():
        if stats["type"] == "API":
            avg_input = stats["input"]
            avg_output = stats["output"]
            pricing = api_model_pricing[model]
            cost_per_prompt = avg_input * (pricing["input"] / 1000) + avg_output * (
                pricing["output"] / 1000
            )
            final_results[benchmark][model] = {
                "type": "API",
                "cost_per_prompt": cost_per_prompt,
            }
        else:
            n_samples = len(stats["costs"])
            avg = sum(stats["costs"]) / n_samples
            final_results[benchmark][model] = {
                "type": "Open Source",
                "avg_cost_per_prompt": avg,
            }

print(final_results)
