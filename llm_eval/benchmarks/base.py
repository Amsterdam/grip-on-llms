"""
Module for handling benchmarks and running evaluation.
For every benchmark we should be able to provide an LLM,
generate LLM responses and evaluate them.
"""
import json
from abc import ABC, abstractmethod


class BaseBenchmark(ABC):
    """Base benchmark class"""

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """Property to get the benchmark name"""
        return self._name

    def run(self, llm, results_path=None):
        """Run the benchmark using the provided LLM."""
        results = self._run_task(llm)

        if results_path:
            with open(results_path, "w") as f:
                json.dump(results, f)

        return results

    def score(self, results=None, results_path=None):
        """Given results or a path to results, calculate desired score"""
        if not results and not results_path:
            raise ValueError("At least one of results or results path must be provided")

        if results_path:
            results = json.load(open(results_path, "r"))

        score = self._calculate_metric(results)

        return score

    def eval(self, llm, results_path=None):
        """Run benchmark and calculate corresponding scores"""
        run_output = self.run(llm)
        score = self.score(run_output)

        results = {
            "run_output": run_output,
            "score": score,
        }

        if results_path:
            with open(results_path, "w") as f:
                json.dump(results, f)

        return results

    @abstractmethod
    def _run_task(self, llm):
        """Function to run a task should always be implemented"""
        raise NotImplementedError("Implement _run_task function")

    @abstractmethod
    def _calculate_metric(self, results):
        """Function to calculate a metric should always be implemented"""
        raise NotImplementedError("Implement _calculate_metric function")

    def get_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "name": self.name,
        }
        metadata.update(self._get_own_metadata())
        return metadata

    @abstractmethod
    def _get_own_metadata(self):
        """Get benchmark-specific metadata for versioning purposes"""
        raise NotImplementedError("Implement _get_own_metadata function")
