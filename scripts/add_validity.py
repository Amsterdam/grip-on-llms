"""Retroactively add validity to all results"""
import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

# isort off to ensure that paths are set up before transformers imports
# isort: off
from llm_eval.utils.setup_utils import (
    base_results_folder,
    benchmark_data_folder,
    get_gpt_secrets,
    get_hf_secrets,
)

# isort: on

from llm_eval.benchmarks import (
    ARC,
    MMLU,
    AmsterdamSimplification,
    BZKSocialBias,
    CNNDailyMail,
    DutchBBQ,
    DutchCrowSPairs,
    HonestCityBench,
    INTDuidelijkeTaal,
    TinyARC,
    TinyMMLU,
    TinyTruthfulQA,
    XSum,
)
from llm_eval.language_models import LLMRouter
from llm_eval.translators import TranslatorRouter
from llm_eval.utils.schemas import BenchmarkResult

gpt_secrets = get_gpt_secrets()
hf_secrets = get_hf_secrets()

CODE_CARBON_PARAMS = {
    "country_iso_code": "SWE",
    "region": "sweden",
    "allow_multiple_runs": True,
    "save_to_file": False,
    "pue": 1.185,
    # Directly set log level as suppressing logs doesn't work in other ways
    "log_level": "WARNING",
}

GPT_TRANSLATION_PARAMS = {
    # temp ~0-0.2 to avoid weird translations
    "temperature": 0.2,
    # top_p ~0.95-1
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
}

GPT_PARAMS = {
    "temperature": 0,
    "top_p": 1,
    # Don't penalize to ensure greedy decoding
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    # "stop": None,
    "max_tokens": 200,
}

# gpt-5 models use 200 tokens for reasoning => increasing max tokens or turn off reasoning
GPT_5_PARAMS = {
    # "temperature": 0,
    "top_p": 1,
    # Don't penalize to ensure greedy decoding
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    # "stop": None,
    "max_completion_tokens": 200,
    "reasoning_effort": "minimal",
}

HF_INFERENCE_PARAMS = {
    "do_sample": False,
    # temp, top_k & top_p - unused for greedy decoding (adding for transparency)
    "temperature": 0,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "num_return_sequences": 1,
    # "no_repeat_ngram_size": 3,
    "max_new_tokens": 200,
}

HF_OBJECT_PARAMS = {
    "provider": "vllm",
    # "provider": "huggingface",
    "hf_token": hf_secrets["HF_TOKEN"],
    "params": HF_INFERENCE_PARAMS,
    "uses_api": False,
}

HF_MODELS = [
    "mistral-7b-instruct-v0.3",
    "mistral-small-instruct",
    "tiny-llama",
    "llama-3.1-8b-instruct",
    "llama-3.3-70b-gptq",
    "fietje-2-instruct",
    "geitje-7b-ultra",
    "phi-4-mini-instruct",
    "falcon3-7b-instruct",
    "olmo-7b-instruct",
    "olmo-32b-instruct",
    "eurollm-9b-instruct",
    "eurollm-22b-instruct",
    "qwen-8b",
    "qwen-32b",
    "qwen3-32b-awq",
    "gemma-12b-instruct",
    "gemma-27b-instruct",
    "smollm3-3b",
    "aya-expanse-32b",
    "command-r7b",
    "apertus-8b-instruct",
    "apertus-70b-instruct-quantized",
    "gpt-oss-20b",
    # "gpt-oss-120b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-70b-awq",
]

PRIVATE_AZURE_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
]

NON_PRIVATE_AZURE_MODELS = [
    "o1-mini",
    "Llama-3-3-70B-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
]

AZURE_MODELS = PRIVATE_AZURE_MODELS + NON_PRIVATE_AZURE_MODELS

EXISTING_BENCHMARKS = [
    # Simplification
    "AmsterdamSimplification",
    "INT_Duidelijke_Taal",
    # Summarization
    "CNNDailyMail",
    "XSum",
    # Honesty
    "HonestCity",
    # Factuality
    # "MMLU-NL",
    # "ARC-NL",
    "TinyMMLU",
    "TinyARC",
    "TinyTruthfulQA",
    # Bias
    "Dutch-BBQ",
    "Dutch-CrowSPairs",
    "BZK-Social-Bias-gender",
    "BZK-Social-Bias-name",
]

translation_gpt = LLMRouter.get_model(
    provider="azure",
    model_name="gpt-4o",
    api_endpoint=gpt_secrets["API_ENDPOINT"],
    api_key=gpt_secrets["API_KEY"],
    api_version=gpt_secrets["API_VERSION"],
    params=GPT_TRANSLATION_PARAMS,
    uses_api=True,
)


def get_model(model_name, params=None):
    logging.info(f"Initializing {model_name}")
    if model_name in AZURE_MODELS:
        params = params or (GPT_PARAMS if "gpt-4o" in model_name else GPT_5_PARAMS)
        print(params)
        return LLMRouter.get_model(
            provider="azure",
            model_name=model_name,
            api_endpoint=gpt_secrets["API_ENDPOINT"],
            api_key=gpt_secrets["API_KEY"],
            api_version=gpt_secrets["API_VERSION"],
            params=params,
            uses_api=True,
        )
    elif model_name in HF_MODELS:
        params = params or HF_OBJECT_PARAMS
        return LLMRouter.get_model(
            model_name=model_name,
            **params,
        )
    else:
        raise ValueError("Unknown model")


BENCH_CLASSES = {
    "TinyMMLU": TinyMMLU,
    "TinyARC": TinyARC,
    "TinyTruthfulQA": TinyTruthfulQA,
    "MMLU-NL": MMLU,
    "ARC-NL": ARC,
    "INT_Duidelijke_Taal": INTDuidelijkeTaal,
    "AmsterdamSimplification": AmsterdamSimplification,
    "CNNDailyMail": CNNDailyMail,
    "XSum": XSum,
    "Dutch-BBQ": DutchBBQ,
    "Dutch-CrowSPairs": DutchCrowSPairs,
}

BENCH_FILES = {
    "MMLU-NL": "mmmlu_nl_dev.json",
    "ARC-NL": "marc_nl_validation.json",
    "INT_Duidelijke_Taal": "CrowdsourcingResults.csv",
    "AmsterdamSimplification": "complex-simple-v1-anonymized.csv",
}


def get_tiny_bench(bench_name, bench_language, translator):
    bench_class = BENCH_CLASSES[bench_name]
    data_dir = Path(benchmark_data_folder) / bench_name
    return bench_class(
        benchmark_name=bench_name,
        language=bench_language,
        data_dir=data_dir,
        translator=translator,
    )


def get_fact_bench(bench_name):
    bench_class = BENCH_CLASSES[bench_name]
    bench_file = BENCH_FILES[bench_name]
    data_path = Path(benchmark_data_folder) / bench_name / bench_file
    # MMLU(benchmark_name, data_path=data_path, categories=["moral_disputes"])
    # ARC(benchmark_name, data_path=data_path, categories=["LEAP"])
    return bench_class(bench_name, data_path=data_path, categories=[])


def get_simple_bench(bench_name, simple_prompt_type):
    bench_class = BENCH_CLASSES[bench_name]
    bench_file = BENCH_FILES[bench_name]
    data_path = Path(benchmark_data_folder) / bench_name / bench_file
    return bench_class(
        benchmark_name=f"{bench_name}-{simple_prompt_type}",
        data_path=data_path,
        prompt_type=simple_prompt_type,
    )


def get_summary_bench(
    bench_name, bench_language, summary_prompt_type, translator, max_translation_entries
):
    bench_class = BENCH_CLASSES[bench_name]
    data_dir = Path(benchmark_data_folder) / bench_name
    return bench_class(
        benchmark_name=bench_name,
        language=bench_language,
        prompt_type=summary_prompt_type,
        data_dir=data_dir,
        translator=translator,
        max_translation_entries=max_translation_entries,
    )


def get_dutch_bias_bench(bench_name):
    bench_class = BENCH_CLASSES[bench_name]
    data_dir = Path(benchmark_data_folder) / bench_name
    return bench_class(
        benchmark_name=bench_name,
        language="NL",
        data_dir=data_dir,
    )


def get_benchmark(
    bench_name,
    simple_prompt_type=None,
    summary_prompt_type=None,
    bench_language="NL",
    max_translation_entries=None,
    translator=None,
):
    # Run mmlu or arc using the local dumps
    if bench_name in ["MMLU-NL", "ARC-NL"]:
        return get_fact_bench(bench_name)

    elif bench_name in ["INT_Duidelijke_Taal", "AmsterdamSimplification"]:
        return get_simple_bench(bench_name, simple_prompt_type)

    elif bench_name in ["CNNDailyMail", "XSum"]:
        return get_summary_bench(
            bench_name, bench_language, summary_prompt_type, translator, max_translation_entries
        )

    elif bench_name in ["TinyMMLU", "TinyARC", "TinyTruthfulQA"]:
        return get_tiny_bench(bench_name, bench_language, translator)

    elif bench_name in ["Dutch-BBQ", "Dutch-CrowSPairs"]:
        return get_dutch_bias_bench(bench_name)

    elif bench_name.startswith("BZK-Social-Bias"):
        aspect = bench_name.split("-")[-1]
        return BZKSocialBias(
            which_test=aspect,
            benchmark_name="BZK-Social-Bias",
            language="NL",
            data_dir=Path(benchmark_data_folder) / f"BZK-Social-Bias-{aspect}",
        )

    elif bench_name == "HonestCity":
        data_path = Path(benchmark_data_folder) / bench_name / "honest_city_final_annotated.xlsx"
        return HonestCityBench(
            bench_name,
            data_path=data_path,
            llm_judges=[],
        )

    else:
        raise ValueError(f"Benchmark {bench_name} unknown.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    default_experiment = "2025-10-03-rerun-new-schema"
    parser.add_argument(
        "experiment_name",
        type=str,
        nargs="?",  # Makes it optional
        default=default_experiment,
        help=f"Name of the experiment (default: {default_experiment})",
    )

    parser.add_argument(
        "--benches",
        nargs="+",
        default=EXISTING_BENCHMARKS,
        help=f"Benchmarks to run (default/existing: {EXISTING_BENCHMARKS})",
    )

    # default_models = HF_MODELS + AZURE_MODELS
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="List of models to evaluate",
    )
    parser.add_argument("--private_data", action="store_true", help="Only run compliant models.")
    parser.add_argument(
        "--only_self_hosted", action="store_true", help="Only run self-hosted models."
    )
    parser.add_argument("--only_azure", action="store_true", help="Only run azure models.")

    default_n_samples = 15000
    parser.add_argument(
        "--n-samples",
        type=int,
        default=default_n_samples,
        help=f"Number of samples to use (default: {default_n_samples})",
    )

    max_translation_entries = 100
    parser.add_argument(
        "--max_translation_entries",
        type=int,
        default=max_translation_entries,
        help=f"Max entries when translating benchmarks (default: {max_translation_entries})",
    )

    parser.add_argument(
        "--summary_prompt_type",
        type=str,
        default="detailed",
        help="Prompt type for summary benchmarks - (default: detailed)",
    )

    parser.add_argument(
        "--simple_prompt_type",
        type=str,
        default="detailed",
        help="Prompt type for simplification benchmarks - (default: detailed)",
    )

    parser.add_argument(
        "--language", type=str, default="NL", help="Benchmark language (default: NL)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="To be implemented: read results if existing and force rerun if desired.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":  # noqa: C901
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)

    translation_model = get_model("gpt-4o", params=GPT_TRANSLATION_PARAMS)
    en_nl_translator = TranslatorRouter.get_translator(
        translator_type="llm_based",
        model_name=translation_model.model_name,
        llm=translation_model,
        source_lang="EN",
        target_lang="NL",
    )

    for bench_name in args.benches:
        logging.info(f"Processing {bench_name}")

        bench = get_benchmark(
            bench_name,
            simple_prompt_type=args.simple_prompt_type,
            summary_prompt_type=args.summary_prompt_type,
            bench_language=args.language,
            max_translation_entries=args.max_translation_entries,
            translator=en_nl_translator,
        )

        results_dir = Path(base_results_folder) / args.experiment_name / bench.name

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
                result = BenchmarkResult.load(results_dir / results_file)

                metadata = result.metadata
                benchmark = metadata.benchmark.name
                model_name = metadata.llm.model_name
                run_output = result.run_output

                if args.models and model_name not in args.models:
                    logging.info(f"Skipping {model_name}")
                    continue

                if result.validity:
                    logging.info(f"Skipping {model_name} (validity exists)")
                    skipped += 1
                    continue

                if args.dry_run:
                    logging.info(f"Could have processed: {model_name}")
                    continue

                logging.info(f"Processing {model_name}")
                responses = result.run_output
                metrics = result.evaluation.metrics
                validity = bench.check_validity(responses, metrics)
                result.validity = validity
                result.save(results_dir / results_file)

                processed += 1

            except Exception as e:
                logging.error(f"Error processing {results_file}: {e}")
                errors += 1

        logging.info(
            f"Done with {bench}! {processed} processed, {skipped} skipped, {errors} errors"
        )
