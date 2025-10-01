"""Add or regenerated honesty judgements for a given experiment"""
import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

# isort: off
from llm_eval.utils.setup_utils import (
    base_results_folder,
    get_gpt_secrets,
    get_hf_secrets,
)

# isort: on

from llm_eval.benchmarks.honesty.honest_city_eval import HonestCityEvaluator
from llm_eval.language_models import LLMRouter
from llm_eval.utils.schemas import BenchmarkResult


def get_judges(judge_names, gpt_secrets, hf_secrets):
    """Initialize judge models based on names"""
    logging.info(f"Getting judges: {judge_names}")
    gpt_judge_params = {
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "max_tokens": 20,
    }

    hf_judge_infenerence_params = {
        "do_sample": False,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "num_return_sequences": 1,
        "max_new_tokens": 20,
    }

    hf_judge_object_params = {
        # "provider": "vllm",
        "provider": "huggingface",
        "hf_token": hf_secrets["HF_TOKEN"],
        "params": hf_judge_infenerence_params,
        "uses_api": False,
    }

    judges = []

    for name in judge_names:
        if name == "gpt-4o-mini":
            judges.append(
                LLMRouter.get_model(
                    provider="azure",
                    model_name="gpt-4o-mini",
                    api_endpoint=gpt_secrets["API_ENDPOINT"],
                    api_key=gpt_secrets["API_KEY"],
                    api_version=gpt_secrets["API_VERSION"],
                    params=gpt_judge_params,
                    uses_api=True,
                )
            )

        elif name == "qwen-8b":
            judges.append(
                LLMRouter.get_model(
                    model_name="qwen-8b",
                    **hf_judge_object_params,
                )
            )

        elif name == "gemma-12b-instruct":
            judges.append(
                LLMRouter.get_model(
                    model_name="gemma-12b-instruct",
                    **hf_judge_object_params,
                )
            )

        elif name == "mistral-7b-instruct":
            judges.append(
                LLMRouter.get_model(
                    model_name="mistral-7b-instruct-v0.3",
                    **hf_judge_object_params,
                )
            )

        elif name == "tiny-llama":
            judges.append(
                LLMRouter.get_model(
                    model_name="tiny-llama",
                    **hf_judge_object_params,
                )
            )

        else:
            raise ValueError("Unsupported judge model")

    return judges


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name", type=str)
    parser.add_argument(
        "--judges", nargs="+", default=["gpt-4o-mini", "qwen-8b", "gemma-12b-instruct"]
    )
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

    HONESTY_FOLDER = "HonestCity"
    results_dir = Path(base_results_folder) / args.experiment_name / HONESTY_FOLDER

    if not results_dir.exists():
        logging.error(f"Directory not found: {results_dir}")
        exit(1)

    results_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    logging.info(f"Found {len(results_files)} result files: {results_files}")

    gpt_secrets = get_gpt_secrets()
    hf_secrets = get_hf_secrets()
    judges = get_judges(args.judges, gpt_secrets, hf_secrets)
    judge_names = [judge.model_name for judge in judges]
    evaluator = HonestCityEvaluator(judge_llms=judges)

    processed = 0
    skipped = 0
    errors = 0

    for results_file in tqdm(results_files):
        try:
            # entry = json.loads(open(results_dir / results_file, "rb").read())
            result = BenchmarkResult.load(results_dir / results_file)

            metadata = result.metadata
            benchmark = metadata.benchmark.name
            model_name = metadata.llm.model_name

            if benchmark != "HonestCity":
                logging.warning(f"Glitch! Found a {benchmark} file.")
                continue

            if (
                result.evaluation is not None
                and set(result.evaluation.eval_metadata.judges) == set(judge_names)
                and not args.force
            ):
                logging.info(f"Skipping {model_name} (already evaluated)")
                skipped += 1
                continue

            if args.dry_run:
                logging.info(f"Could have processed: {model_name}")
                continue

            logging.info(f"Processing {model_name}")
            responses = result.run_output
            evaluation_results = evaluator.evaluate(responses, force=args.force)
            result.evaluation = evaluation_results
            result.save(results_dir / results_file)

            processed += 1

        except Exception as e:
            logging.error(f"Error processing {results_file}: {e}")
            errors += 1

    logging.info(f"Done! {processed} processed, {skipped} skipped, {errors} errors")
