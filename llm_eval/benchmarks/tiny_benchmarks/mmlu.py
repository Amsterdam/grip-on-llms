"""
Implementation of the tiny version of the MMLU benchmark which contains
57 tasks and ims to measure world knowledge and problem solving ability [1].

We directly use the formatted inputs from the TinyBenchmarks [2]
available in the huggingface hub
(https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU).

References:
[1] Hendrycks, Dan, et al. "Measuring massive multitask language understanding."
arXiv preprint arXiv:2009.03300 (2020).

[2] Polo, Felipe Maia, et al.
"tinyBenchmarks: evaluating LLMs with fewer examples."
arXiv preprint arXiv:2402.14992 (2024).
"""
from datasets import load_dataset

from llm_eval.benchmarks.tiny_benchmarks.base import BaseTinyBenchmark

ANSWERS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

BENCHMARK_PURPOSE = (
    "The purpose of the benchmark is to measure world knowledge "
    "and problem solving ability by answering multiple-choice questions."
)


class TinyMMLU(BaseTinyBenchmark):
    """TinyMMLU implementation."""

    def __init__(
        self,
        benchmark_name,
        language="NL",
        data_dir=None,
        translator=None,
        max_translation_entries=10,
    ):
        """Initialize XSum benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            input_field="input_formatted",
            target_field="answer",
            benchmark_purpose=BENCHMARK_PURPOSE,
            hf_repository="tinyBenchmarks/tinyMMLU",
            data_dir=data_dir,
            language=language,
            translator=translator,
            max_translation_entries=max_translation_entries,
        )

    def _load_huggingface_data(self):
        dataset = load_dataset(self.hf_repository, trust_remote_code=True, split="test")
        return dataset

    def _get_inputs(self):
        """Access source text (full document)"""
        return self.dataset["input_formatted"]

    def _get_targets(self):
        """Access target summary (ground-truth)"""
        return [ANSWERS[x] for x in self.dataset["answer"]]

    def _calculate_metric(self, results=None):
        """Given results, calculate desired score"""
        accuracy = len(
            [
                entry
                for entry in results
                if entry["response"].strip().lower() == entry["target"].strip().lower()
            ]
        ) / len(results)
        return {"acc": accuracy}
