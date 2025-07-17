"""Generate the data for the leaderboard"""
import json
import logging
from datetime import datetime

from tqdm import tqdm

from llm_eval.utils.metadata import get_device_info, get_environment_info


class Leaderboard:
    """Run benchmarks for a number of models and generate the data to be presented"""

    def __init__(self, llms, benchmarks, codecarbon_params, n_samples=0):
        """
        Args:
            llms (list): List of LLMs.
            benchmarks (list): List of Benchmarks.
        (assumes the already initialized LLM/Benchmark instances)
        """
        self.llms = llms
        self.benchmarks = benchmarks
        self.codecarbon_params = codecarbon_params
        self.n_samples = n_samples

    def run_comparison(self, results_path=None):
        """
        For run the full comparison for the provided benchmarks and LLMs.
        Add all necessary metadata, such as LLM & benchmark info, system info,
        timesteps, etc.
        Dump results if a path has been provided.
        """
        datetime_format = "%Y-%m-%dT%H:%M:%SZ"

        results = []
        for llm in tqdm(self.llms, desc="LLMs"):
            # Warm up LLM: temp fix for duration discrepancy
            llm.prompt("Let's benchmark some models!!")

            for benchmark in tqdm(self.benchmarks, desc="Benchmarks"):
                try:
                    self.codecarbon_params["project_name"] = f"{benchmark.name}-{llm.model_name}"
                    llm.initialize_carbon_tracking(self.codecarbon_params)

                    start_time = datetime.now()
                    benchmark_results = benchmark.eval(llm, n_samples=self.n_samples)

                    end_time = datetime.now()

                    results.append(
                        {
                            "metadata": {
                                "llm": llm.get_metadata(),
                                "benchmark": benchmark.get_metadata(),
                                "n_samples": self.n_samples,
                                "run": {
                                    "timestamp": datetime.now().strftime(datetime_format),
                                    "timestamp_bench_start": start_time.strftime(datetime_format),
                                    "timestamp_bench_end": end_time.strftime(datetime_format),
                                    "time_bench_total": str(end_time - start_time),
                                    "system": get_system_metadata(),
                                },
                                "code_carbon": llm.get_carbon_data(),
                            },
                            "benchmark_results": benchmark_results,
                        }
                    )

                    if results_path:
                        with open(results_path + "_tmp", "a") as f:
                            json.dump(results, f, default=str)

                except Exception as e:
                    logging.error(f"{llm.model_name} failed: {e}")

            llm.unload_model()

        if results_path:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4, default=str)

        return results


def get_system_metadata():
    metadata = {
        "environment_info": get_environment_info(),
        "device_info": get_device_info(),
    }
    return metadata
