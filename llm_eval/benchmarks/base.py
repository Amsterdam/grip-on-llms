"""
Module for handling benchmarks and running evaluation.
For every benchmark we should be able to provide an LLM,
generate LLM responses and evaluate them.
"""
import json
from abc import ABC, abstractmethod
from datetime import datetime


class BaseBenchmark(ABC):
    """Base LLM class"""

    def run(self, llm, results_path=None):
        """Run the benchmark using the provided LLM."""
        benchmark_results = self._run_task(llm)

        results = {
            "metadata": {
                "llm": llm.get_metadata(),
                "benchmark": None,
                "run": {"timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")},
            },
            "benchmark_results": benchmark_results,
        }

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
        results = self.run(llm)
        score = self.score(results["benchmark_results"])

        if results_path:
            results["score"] = score
            with open(results_path, "w") as f:
                json.dump(results, f)

        return score

    @abstractmethod
    def _run_task(self, llm):
        """Function to run a task should always be implemented"""
        raise NotImplementedError("Implement _run_task function")

    @abstractmethod
    def _calculate_metric(self, results):
        """Function to calculate a metric should always be implemented"""
        raise NotImplementedError("Implement _calculate_metric function")
