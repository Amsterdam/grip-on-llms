"""Generate the data for the LLM Overview"""
import logging
from datetime import datetime
from pathlib import Path
from pprint import pprint

from tqdm import tqdm

from llm_eval.utils.metadata import get_device_info, get_environment_info
from llm_eval.utils.schemas import BenchmarkResult, MetadataContainer, RunMetadata

DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class LLMOverview:
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

    def run_comparison(self, results_dir=None, force=False):
        """
        For run the full comparison for the provided benchmarks and LLMs.
        Add all necessary metadata, such as LLM & benchmark info, system info,
        timesteps, etc.
        Dump results if a path has been provided.
        """
        for llm in tqdm(self.llms, desc="LLMs"):
            logging.info(f"Evaluating {llm.model_name}")
            try:
                # Warm up LLM: temp fix for duration discrepancy
                llm.prompt("Let's benchmark some models!!")

                for benchmark in tqdm(self.benchmarks, desc="Benchmarks"):
                    if self._should_skip_bench(results_dir, llm, benchmark, force):
                        continue

                    try:
                        self._generate_bench_results(llm, benchmark, results_dir)
                    except Exception as bench_e:
                        logging.error(f"{llm.model_name} failed on {benchmark.name}: {bench_e}")

            except Exception as llm_e:
                logging.error(f"{llm.model_name} failed: {llm_e}")

            finally:
                llm.unload_model()

    def _get_path(self, results_dir, llm, benchmark):
        return Path(results_dir) / benchmark.name / f"{llm.model_name}.json"

    def _generate_bench_results(self, llm, benchmark, results_dir):
        """Generate results for a specific llm and benchmark"""
        self.codecarbon_params["project_name"] = f"{benchmark.name}-{llm.model_name}"
        llm.initialize_carbon_tracking(self.codecarbon_params)

        start_time = datetime.now()
        run_output, evaluation, validity = benchmark.eval(llm, n_samples=self.n_samples)

        end_time = datetime.now()

        result = BenchmarkResult(
            metadata=MetadataContainer(
                llm=llm.get_metadata(),
                benchmark=benchmark.get_metadata(),
                run=RunMetadata(
                    timestamp=datetime.now().strftime(DATETIME_FORMAT),
                    timestamp_bench_start=start_time.strftime(DATETIME_FORMAT),
                    timestamp_bench_end=end_time.strftime(DATETIME_FORMAT),
                    time_bench_total=str(end_time - start_time),
                    system=get_system_metadata(),
                ),
                code_carbon=llm.get_carbon_data(),
                n_samples=self.n_samples,
            ),
            run_output=run_output,
            evaluation=evaluation,
            validity=validity,
        )

        if results_dir:
            results_path = self._get_path(results_dir, llm, benchmark)
            result.save(results_path)

        return result

    def _should_skip_bench(self, results_dir, llm, benchmark, force):
        """Check whether it's possible to load results and whether they are valid"""
        if force or not results_dir:
            return False

        results_path = self._get_path(results_dir, llm, benchmark)
        llm_bench_names = f"{llm.model_name} {benchmark.name}"

        try:
            results = BenchmarkResult.load(results_path)
        except Exception as e:
            logging.info(f"Couldn't load results for {llm_bench_names}: {e}")
            return False

        if not results.validity:
            logging.info(f"No validity information for {llm_bench_names}")
            return False

        pprint(results.validity)
        if results.validity["is_valid"]:
            logging.info(f"Reusing results for {llm_bench_names}")
            return True
        else:
            logging.info(f"Found invalid results for {llm_bench_names}")
            return False


def get_system_metadata():
    metadata = {
        "environment_info": get_environment_info(),
        "device_info": get_device_info(),
    }
    return metadata
