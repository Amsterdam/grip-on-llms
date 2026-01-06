"""Add or regenerated honesty judgements for a given experiment"""
import argparse
import logging
import os
from pathlib import Path

import tiktoken
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from tqdm import tqdm

# isort: off
from llm_eval.utils.setup_utils import base_results_folder

# isort: on

from llm_eval.utils.schemas import BenchCosts, BenchmarkResult

# API pricing table (€ per 1k tokens as of 6 November 2025)
# considering an ECB exchange rate of 1 USD = 0.8702 EUR
API_MODEL_PRICING = {
    # $0.0003 / $0.0001 per 1000
    "Mistral-small-2503": {"input": 0.00026106, "output": 0.00008702},
    # $0.0004 / $0.002 per 1000
    "Mistral-medium-2505": {"input": 0.00034808, "output": 0.0017404},
    # $0.0005 / $0.0015 per 1000
    "Mistral-Large-3": {"input": 0.0004351, "output": 0.0013053},
    # $2.50 / $10.00
    "gpt-4o": {"input": 0.0021755, "output": 0.008702},
    # $0.15 / $0.60
    "gpt-4o-mini": {"input": 0.00013053, "output": 0.00052212},
    # $1.25 / $10.00
    "gpt-5": {"input": 0.00108775, "output": 0.008702},
    # $0.25 / $2.00
    "gpt-5-mini": {"input": 0.00021755, "output": 0.0017404},
    # $0.05 / $0.40
    "gpt-5-nano": {"input": 0.00004351, "output": 0.00034808},
    # $1.10	/ $4.40
    "o1-mini": {"input": 0.00095722, "output": 0.00382888},
    # as of 14 November 2025
    # https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/llama/
    "Llama-3-3-70B-Instruct": {"input": 0.000670, "output": 0.000670},
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {"input": 0.000236, "output": 0.000944},
}

# Current GPU rates on Azure (in € per hour as of 26 August 2025,
# considering an exchange rate of 1 USD = 0.8679 EUR
GPU_HOURLY_RATES = {
    "Tesla T4": 0.5728,
    "NVIDIA H100 NVL": 7.8805,
    "Tesla V100-PCIE-16GB": 3.3154,
}


def count_tokens(model_name, text):
    """Count tokens using tiktoken for a given model and text."""
    try:
        if "Mistral" in model_name:
            tokenizer = MistralTokenizer.v3().instruct_tokenizer.tokenizer
            tokens = tokenizer.encode(text, bos=True, eos=False)
            return len(tokens)
        else:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
    except Exception as e:
        print(f"Couldn't count tokens for {model_name}: {e}")
        return 0


def get_input_text(entry):
    if hasattr(entry, "formatted_prompt") and entry.formatted_prompt:
        if isinstance(entry.formatted_prompt, list):
            return entry.formatted_prompt[0].get("content", "")
    return entry.prompt or ""


def calculate_api_costs(results):
    try:
        metadata = results.metadata
        model_name = metadata.llm.model_name
        bench_name = metadata.benchmark.name
        pricing = API_MODEL_PRICING[model_name]

        # Assuming single entry in formatted prompt?
        inputs = [get_input_text(entry) for entry in results.run_output]
        outputs = [entry.raw_response for entry in results.run_output]
        n_input = sum(
            count_tokens(model_name, input) for input in inputs if isinstance(input, str)
        )
        n_output = sum(
            count_tokens(model_name, output) for output in outputs if isinstance(output, str)
        )

        # consider pricing per 1000 tokens
        total_costs = n_input * pricing["input"] / 1000 + n_output * pricing["output"] / 1000

        costs = BenchCosts(
            total_cost=total_costs,
            cost_per_prompt=total_costs / results.evaluation.total_samples,
            n_samples=results.evaluation.total_samples,
            method="api",
            error=False,
            api_pricing=pricing,
            token_counts={"input": n_input, "output": n_output},
        )

    except Exception as e:
        logging.error(f"Costs calculation for {model_name} & {bench_name} failed: {e}")
        costs = BenchCosts(method="api", error=True, exception=str(e))

    return costs


def calculate_gpu_costs(results):
    try:
        metadata = results.metadata
        model_name = metadata.llm.model_name
        bench_name = metadata.benchmark.name
        system_metadata = metadata.run.system
        device = system_metadata["device_info"]["gpu"]["device_name"]

        hourly_rate = GPU_HOURLY_RATES[device]
        duration = metadata.code_carbon["duration"]
        total_costs = duration * hourly_rate / 3600

        costs = BenchCosts(
            total_cost=total_costs,
            cost_per_prompt=total_costs / results.evaluation.total_samples,
            n_samples=results.evaluation.total_samples,
            method="gpu",
            error=False,
            gpu_type=device,
            duration_seconds=duration,
            gpu_hourly_rate=hourly_rate,
        )

    except Exception as e:
        logging.error(f"Costs calculation for {model_name} & {bench_name} failed: {e}")
        costs = BenchCosts(method="gpu", error=True, exception=str(e))

    return costs


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--benches",
        nargs="+",
        default=[
            "AmsterdamSimplification-detailed",
            "INT_Duidelijke_Taal-detailed",
            "CNNDailyMail",
            "XSum",
            "HonestCity",
        ],
    )
    parser.add_argument("--models", nargs="+", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate even if results already exist"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":  # noqa: C901
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    for bench in args.benches:
        logging.info(f"Processing {bench}")

        results_dir = Path(base_results_folder) / args.experiment_name / bench

        if not results_dir.exists():
            logging.error(f"Directory not found: {results_dir}")
            exit(1)

        results_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
        logging.info(f"Found {len(results_files)} result files: {results_files}")

        processed = 0
        skipped = 0
        errors = 0

        for results_file in tqdm(results_files):
            try:
                results = BenchmarkResult.load(results_dir / results_file)

                metadata = results.metadata
                benchmark = metadata.benchmark.name
                model_name = metadata.llm.model_name

                if benchmark != bench:
                    logging.warning(f"Glitch! Found a {benchmark} file in {bench} folder.")
                    continue

                if args.models and model_name not in args.models:
                    logging.info(f"Skipping {model_name}")
                    continue

                if results.costs is not None and not args.force:
                    logging.info(f"Skipping {model_name} (already evaluated)")
                    skipped += 1
                    continue

                if args.dry_run:
                    logging.info(f"Could have processed: {model_name}")
                    continue

                logging.info(f"Processing {model_name}")
                if model_name in API_MODEL_PRICING:
                    costs = calculate_api_costs(results)
                else:
                    costs = calculate_gpu_costs(results)

                results.costs = costs
                results.save(results_dir / results_file)

                processed += 1

            except Exception as e:
                logging.error(f"Error processing {results_file}: {e}")
                errors += 1

        logging.info(
            f"Done with {bench}! {processed} processed, {skipped} skipped, {errors} errors"
        )
