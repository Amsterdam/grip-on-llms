"""
Implementation of the tiny version of the AI2’s Reasoning Challenge (ARC) benchmark
which measures common sense reasing and contains multiple-choice questions
from science exams from grade 3 to grade 9 [1].

We directly use the formatted inputs from the TinyBenchmarks [2]
available in the huggingface hub
(https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU).

References:
[1] Clark, Peter, et al.
"Think you have solved question answering? try arc, the ai2 reasoning challenge."
arXiv preprint arXiv:1803.05457 (2018).

[2] Polo, Felipe Maia, et al.
"tinyBenchmarks: evaluating LLMs with fewer examples."
arXiv preprint arXiv:2402.14992 (2024).
"""
from datasets import load_dataset

from llm_eval.benchmarks.tiny_benchmarks.base_mc import (
    BaseTinyMultipleChoiceBenchmark,
    template_question,
)

BENCHMARK_PURPOSE = (
    "The purpose of the benchmark is to measure common sense reasing"
    "using multiple-choice questions from science exams"
)


class TinyARC(BaseTinyMultipleChoiceBenchmark):
    """TinyARC implementation."""

    def __init__(
        self,
        benchmark_name,
        language="NL",
        data_dir=None,
        translator=None,
    ):
        """Initialize TinyARC benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            input_field="input_formatted_0shot",
            target_field="answerKey",
            benchmark_purpose=BENCHMARK_PURPOSE,
            hf_repository="tinyBenchmarks/tinyAI2_arc",
            data_dir=data_dir,
            language=language,
            translator=translator,
            tiny_task="arc",
        )

    def _load_huggingface_data(self):
        dataset = load_dataset(self.hf_repository, trust_remote_code=True, split="test")
        dataset = dataset.map(
            lambda data_point: {
                **data_point,
                self.input_field: template_question(
                    data_point["question"], data_point["choices"]["text"]
                ),
            }
        )

        return dataset

    def _get_targets(self):
        """Get ground-truth answers"""
        # Take target directly as it's already A/B/C
        return self.dataset[self.target_field]
