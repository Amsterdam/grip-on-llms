"""Generate the data for the leaderboard"""
import json
from datetime import datetime

from tqdm import tqdm

from llm_eval.utils.metadata import get_device_info, get_environment_info


class Leaderboard:
    """Run benchmarks for a number of models and generate the data to be presented"""

    def __init__(self, llms, benchmarks):
        """
        Args:
            llms (list): List of LLMs.
            benchmarks (list): List of Benchmarks.
        (assumes the already initialized LLM/Benchmark instances)
        """
        self.llms = llms
        self.benchmarks = benchmarks

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
            # llm.start_carbon_tracking()

            for benchmark in tqdm(self.benchmarks, desc="Benchmarks"):
                # llm.(re)start_carbon_tracking()

                start_time = datetime.now()
                benchmark_results = benchmark.eval(llm)
                end_time = datetime.now()

                results.append(
                    {
                        "metadata": {
                            "llm": llm.get_metadata(),
                            "benchmark": benchmark.get_metadata(),
                            "run": {
                                "timestamp": datetime.now().strftime(datetime_format),
                                "timestamp_bench_start": start_time.strftime(datetime_format),
                                "timestamp_bench_end": end_time.strftime(datetime_format),
                                "time_bench_total": str(end_time - start_time),
                                "system": get_system_metadata(),
                            },
                            # "code_carbon": llm.get_carbon_data(),
                        },
                        "benchmark_results": benchmark_results,
                    }
                )

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
