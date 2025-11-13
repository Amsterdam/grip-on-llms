"""
Module for handling benchmarks and running evaluation.
For every benchmark we should be able to provide an LLM,
generate LLM responses and evaluate them.
"""
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import mmh3
import numpy as np

from llm_eval.utils.schemas import BenchmarkEvaluation, BenchmarkMetadata, RunItem


class BaseBenchmark(ABC):
    """Base benchmark class"""

    def __init__(
        self,
        benchmark_name,
        source_url=None,
        data_dir=None,
        data_path=None,
        hf_repository=None,
        preferred_response_format=None,
    ):
        self._name = benchmark_name

        if not source_url and not data_path and not hf_repository:
            raise ValueError(
                "At least one of source_url or data_path or hf_repository must be provided"
            )

        self._source_url = source_url
        self._hf_repository = hf_repository
        self._data_dir = Path(data_dir) if data_dir else Path("./data") / self.name

        self._data_path = (
            Path(data_path) if data_path else self._data_dir / self.name / "data.json"
        )

        self._preferred_response_format = preferred_response_format

    @property
    def name(self):
        """Property to get the benchmark name"""
        return self._name

    @property
    def source_url(self):
        """Property to get the source url"""
        return self._source_url

    @property
    def data_dir(self):
        """Property to get the data dir"""
        return self._data_dir

    @property
    def data_path(self):
        """Property to get the data path"""
        return self._data_path

    @property
    def hf_repository(self):
        """Property to get the hugging face repository"""
        return self._hf_repository

    @property
    def preferred_response_format(self):
        """Property to get the preferred response format"""
        return self._preferred_response_format

    def run(self, llm, n_samples=0) -> List[RunItem]:
        """Run the benchmark using the provided LLM."""
        return self._run_task(llm, n_samples=n_samples)

    def score(self, run_output: List[RunItem]) -> BenchmarkEvaluation:
        """Calculate evaluation from run output"""
        return self._calculate_metrics(run_output)

    def eval(
        self, llm, results_path=None, n_samples=0
    ) -> Tuple[List[RunItem], BenchmarkEvaluation, Dict]:
        """Run benchmark and calculate corresponding scores"""
        run_output = self.run(llm, n_samples=n_samples)
        scores = self.score(run_output)
        validity = self.check_validity(run_output=run_output, scores=scores)
        return run_output, scores, validity

    def check_validity(self, run_output: List[RunItem], scores: BenchmarkEvaluation) -> Dict:
        """Check the validity of run output and scores"""
        valid = [entry for entry in run_output if not entry.error]
        invalid = [entry for entry in run_output if entry.error]
        exceptions = Counter([entry.exception for entry in invalid])
        empty_raw = [
            entry for entry in run_output if not entry.raw_response or len(entry.raw_response) == 0
        ]
        empty_processed = [
            entry
            for entry in run_output
            if not entry.processed_response or len(entry.processed_response) == 0
        ]

        validity = {
            "n_valid_responses": len(valid),
            "valid_responses_rate": len(valid) / len(run_output),
            "n_invalid_responses": len(invalid),
            "invalid_responses_rate": len(invalid) / len(run_output),
            "n_total_responses": len(run_output),
            "n_empty_raw_responses": len(empty_raw),
            "empty_raw_responses_rate": len(empty_raw) / len(run_output),
            "n_empty_processed_responses": len(empty_processed),
            "empty_processed_responses_rate": len(empty_processed) / len(run_output),
            "exceptions": exceptions,
            "is_invalid_reasons": [],
        }

        # check if too many invalid responses
        invalid_response_rate_threshold = 0.25
        if validity["invalid_responses_rate"] >= invalid_response_rate_threshold:
            validity["is_invalid_reasons"].append(
                f"more than {invalid_response_rate_threshold * 100}% invalid responses"
            )

        # check if too many empty processed/raw responses
        empty_response_rate_threshold = 0.25
        if validity["empty_raw_responses_rate"] > empty_response_rate_threshold:
            validity["is_invalid_reasons"].append(
                f"more than {empty_response_rate_threshold * 100}% empty raw responses"
            )
        if validity["empty_processed_responses_rate"] > empty_response_rate_threshold:
            validity["is_invalid_reasons"].append(
                f"more than {empty_response_rate_threshold * 100}% empty processed responses"
            )

        # add bench-specific information;
        # might add bench-specific reasons to invalidate the run
        bench_specific_validity = self._check_validity(run_output=run_output, scores=scores)
        validity["is_invalid_reasons"] += bench_specific_validity.pop("is_invalid_reasons", [])
        validity.update(bench_specific_validity)

        if validity["is_invalid_reasons"]:
            validity["is_valid"] = False
        else:
            validity["is_valid"] = True

        return validity

    @abstractmethod
    def _run_task(self, llm, n_samples=0):
        """Function to run a task should always be implemented"""
        raise NotImplementedError("Implement _run_task function")

    def _sample_data(self, n_samples):
        """
        Get a consistent random sample of the data by hashing elementss,
        sorting them and then taking first n
        """
        data = self._get_hashing_data_for_sampling()
        # ended up using murmurhash because it's supposed to be fast and consistent
        # we could have also done pyhash.xx_64
        hashed_data = [mmh3.hash(entry) for entry in data]
        indices = np.argsort(hashed_data)
        return indices[:n_samples]

    @abstractmethod
    def _get_hashing_data_for_sampling(self):
        """Function to get hashing data for samplig should always be implemented"""
        raise NotImplementedError("Implement _get_hashing_data_for_sampling function")

    @abstractmethod
    def _calculate_metrics(self, results):
        """Function to calculate a metric should always be implemented"""
        raise NotImplementedError("Implement _calculate_metrics function")

    @abstractmethod
    def _check_validity(self, run_output: List[RunItem], scores: BenchmarkEvaluation) -> Dict:
        """Function to check the validity of run output and scores should always be implemented"""
        raise NotImplementedError("Implement _check_validity function")

    def get_metadata(self):
        """Get benchmark metadata for versioning purposes as BenchmarkMetadata object"""
        return BenchmarkMetadata(
            name=self.name,
            source_url=self.source_url,
            data_path=str(self.data_path),
            preferred_response_format=self.preferred_response_format,
            **self._get_own_metadata(),
        )

    @abstractmethod
    def _get_own_metadata(self):
        """Get benchmark-specific metadata for versioning purposes"""
        raise NotImplementedError("Implement _get_own_metadata function")
