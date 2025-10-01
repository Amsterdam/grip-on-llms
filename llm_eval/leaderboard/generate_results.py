"""Generate the data for the leaderboard"""
import logging
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from llm_eval.utils.metadata import get_device_info, get_environment_info
from llm_eval.utils.schemas import BenchmarkResult, MetadataContainer, RunMetadata


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

    def run_comparison(self, results_dir=None):
        """
        For run the full comparison for the provided benchmarks and LLMs.
        Add all necessary metadata, such as LLM & benchmark info, system info,
        timesteps, etc.
        Dump results if a path has been provided.
        """
        datetime_format = "%Y-%m-%dT%H:%M:%SZ"

        results = []
        for llm in tqdm(self.llms, desc="LLMs"):
            try:
                # Warm up LLM: temp fix for duration discrepancy
                llm.prompt("Let's benchmark some models!!")

                for benchmark in tqdm(self.benchmarks, desc="Benchmarks"):
                    try:
                        self.codecarbon_params[
                            "project_name"
                        ] = f"{benchmark.name}-{llm.model_name}"
                        llm.initialize_carbon_tracking(self.codecarbon_params)

                        start_time = datetime.now()
                        run_output, evaluation = benchmark.eval(llm, n_samples=self.n_samples)

                        end_time = datetime.now()

                        result = BenchmarkResult(
                            metadata=MetadataContainer(
                                llm=llm.get_metadata(),
                                benchmark=benchmark.get_metadata(),
                                run=RunMetadata(
                                    timestamp=datetime.now().strftime(datetime_format),
                                    timestamp_bench_start=start_time.strftime(datetime_format),
                                    timestamp_bench_end=end_time.strftime(datetime_format),
                                    time_bench_total=str(end_time - start_time),
                                    system=get_system_metadata(),
                                ),
                                code_carbon=llm.get_carbon_data(),
                                n_samples=self.n_samples,
                            ),
                            run_output=run_output,
                            evaluation=evaluation,
                        )

                        if results_dir:
                            results_path = (
                                Path(results_dir) / benchmark.name / f"{llm.model_name}.json"
                            )
                            result.save(results_path)

                    except Exception as bench_e:
                        logging.error(f"{llm.model_name} failed on {benchmark.name}: {bench_e}")

            except Exception as llm_e:
                logging.error(f"{llm.model_name} failed: {llm_e}")

            finally:
                llm.unload_model()

        return results


def get_system_metadata():
    metadata = {
        "environment_info": get_environment_info(),
        "device_info": get_device_info(),
    }
    return metadata
