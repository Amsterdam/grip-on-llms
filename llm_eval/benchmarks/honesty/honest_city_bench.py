"""
Implementation of HonestCity benchmark.
This benchmark consists of 5 types of prompts.
#TODO: extend on benchmark & methodology. # noqa
"""

import logging
import warnings

import pandas as pd
from tqdm import tqdm

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.benchmarks.honesty.honest_city_eval import HonestCityEvaluator
from llm_eval.utils.exceptions import EmptyResponseError, JudgeMissingWarning


class HonestCityBench(BaseBenchmark):
    """The HonestyCityBench expects..."""

    def __init__(
        self,
        benchmark_name,
        data_dir=None,
        data_path=None,
        hf_repository=None,
        llm_judges=None,
    ):
        """Initialize the benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            data_dir=data_dir,
            data_path=data_path,
            hf_repository=hf_repository,
        )

        self.llm_judges = llm_judges or []

        self._load_data()

    def _load_data(self):
        self.data = pd.read_excel(self.data_path)
        self.data = self.data[self.data["preserve"]]

    def _get_hashing_data_for_sampling(self):
        return [
            f"{entry['category']}-{entry['prompt_cleaned']}-{entry['source']}"
            for _, entry in self.data.iterrows()
        ]

    def _run_task(self, llm, results_path=None, n_samples=0):
        """Run the HonestCityBench using the provided LLM."""
        logging.info(f"Running {self.name} in {n_samples} samples")

        if n_samples:
            indices = self._sample_data(n_samples)
            data = self.data.iloc[indices]
        else:
            data = self.data

        benchmark_results = []

        for _, entry in tqdm(data.iterrows(), desc=f"Running {self.name}"):
            prompt = entry["prompt_cleaned"]

            result = {
                "prompt": prompt,
                "category": entry["category"],
                "source": entry["source"],
            }

            try:
                llm_response = llm.prompt(prompt)
                if not llm_response:
                    raise EmptyResponseError
                result["response"] = llm_response
            except Exception as e:
                result["response"] = ""
                result["error"] = True
                result["exception"] = str(e)

            benchmark_results.append(result)

        return benchmark_results

    def _calculate_metric(self, results=None):
        """Given results, calculate desired score"""
        if self.llm_judges:
            logging.info(f"Calculating Honesty Metrics for {self.name}")
            evaluator = HonestCityEvaluator(self.llm_judges)
            metrics = evaluator.evaluate(results)
            return metrics
        else:
            warnings.warn(
                "HonestCity won't be evaluated; no judges were passed.",
                JudgeMissingWarning,
                stacklevel=2,
            )
            return None

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "data_path": self.data_path,
        }
        return metadata
