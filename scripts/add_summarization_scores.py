"""Add or regenerated honesty judgements for a given experiment"""
import argparse
import logging
import os
from pathlib import Path

from tqdm import tqdm

# isort: off
from llm_eval.utils.setup_utils import base_results_folder

# isort: on

from llm_eval.benchmarks import metrics
from llm_eval.utils.schemas import BenchmarkEvaluation, BenchmarkResult


def calculate_metrics(results, language, required_metrics):
    """Given results, calculate desired score"""
    predictions = [
        entry.processed_response if entry.processed_response else "" for entry in results
    ]
    references = [entry.target if entry.target else "" for entry in results]

    metrics_results = {}

    for metric in required_metrics:
        if metric == "bleu":
            result = metrics.bleu(predictions=predictions, references=references)
        elif metric == "rouge":
            result = metrics.rouge(predictions=predictions, references=references)
        elif metric == "meteor":
            result = metrics.meteor(predictions=predictions, references=references)
        elif metric == "bert_score":
            result = metrics.bertscore(
                predictions=predictions, references=references, lang=language
            )
        else:
            raise ValueError(f"Unknown metric {metric} required")

        metrics_results[metric] = result

    return BenchmarkEvaluation(
        metrics=metrics_results,
        total_samples=len(results),
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_name", type=str)
    parser.add_argument("--metrics", nargs="+", default=["bleu", "rouge", "meteor", "bert_score"])
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

    SUMMARIZATION_BENCHES = ["XSum", "CNNDailyMail"]

    for bench in SUMMARIZATION_BENCHES:
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
                result = BenchmarkResult.load(results_dir / results_file)

                metadata = result.metadata
                benchmark = metadata.benchmark.name
                model_name = metadata.llm.model_name

                if benchmark != bench:
                    logging.warning(f"Glitch! Found a {benchmark} file in {bench} folder.")
                    continue

                if args.models and model_name not in args.models:
                    logging.info(f"Skipping {model_name}")
                    continue

                if (
                    result.evaluation is not None
                    and set(result.evaluation.metrics.keys()) == set(args.metrics)
                    and not args.force
                ):
                    logging.info(f"Skipping {model_name} (already evaluated)")
                    skipped += 1
                    continue

                if args.dry_run:
                    logging.info(f"Could have processed: {model_name}")
                    continue

                logging.info(f"Processing {model_name}")
                language = metadata.benchmark.language
                responses = result.run_output
                evaluation_results = calculate_metrics(responses, language, args.metrics)
                result.evaluation = evaluation_results
                result.save(results_dir / results_file)

                processed += 1

            except Exception as e:
                logging.error(f"Error processing {results_file}: {e}")
                errors += 1

        logging.info(
            f"Done with {bench}! {processed} processed, {skipped} skipped, {errors} errors"
        )
