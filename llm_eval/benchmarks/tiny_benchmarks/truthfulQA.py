"""
Implementation of the tiny version of the TruthfulQA benchmark which
measures whether an LLM is truthful in generating answers to questions [1].
The dataset consists manually crafted questions related to false beliefs
or misconceptions and covers categires such as health, law, finance and politics.

We directly use the formatted inputs from the TinyBenchmarks [2]
available in the huggingface hub
(https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU).

References:
[1] Lin, Stephanie, Jacob Hilton, and Owain Evans.
"Truthfulqa: Measuring how models mimic human falsehoods."
arXiv preprint arXiv:2109.07958 (2021).

[2] Polo, Felipe Maia, et al.
"tinyBenchmarks: evaluating LLMs with fewer examples."
arXiv preprint arXiv:2402.14992 (2024).
"""
import mmh3
from datasets import load_dataset

from llm_eval.benchmarks.tiny_benchmarks.base_mc import (
    BaseTinyMultipleChoiceBenchmark,
    template_question,
)

BENCHMARK_PURPOSE = (
    "The purpose of the benchmark is to measure "
    "whether an LLM is truthful in generating answers to questions"
)


class TinyTruthfulQA(BaseTinyMultipleChoiceBenchmark):
    """TinyTruthfulQA implementation."""

    def __init__(
        self,
        benchmark_name,
        language="NL",
        data_dir=None,
        translator=None,
    ):
        """Initialize TinyTruthfulQA benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            input_field="input_formatted_0shot",
            target_field="answer",
            benchmark_purpose=BENCHMARK_PURPOSE,
            hf_repository="tinyBenchmarks/tinyTruthfulQA",
            data_dir=data_dir,
            language=language,
            translator=translator,
            tiny_task="truthfulqa",
        )

    def _load_huggingface_data(self):
        dataset = load_dataset(self.hf_repository, trust_remote_code=True, split="validation")
        dataset = dataset.map(
            lambda data_point: {
                **data_point,
                "shuffled_mc1_targets": deterministically_shuffle_choices(
                    data_point["mc1_targets"]["choices"],
                    data_point["mc1_targets"]["labels"],
                    max_choices=4,
                ),
            }
        )

        dataset = dataset.map(
            lambda data_point: {
                **data_point,
                self.input_field: template_question(
                    data_point["question"], data_point["shuffled_mc1_targets"]["choices"]
                ),
                self.target_field: data_point["shuffled_mc1_targets"]["label"],
            }
        )

        return dataset


def deterministically_shuffle_choices(choices, labels, max_choices=4):
    # Separate correct and incorrect answers
    # didn't have to bother with multiple, but prep for mc2_targets if needed
    correct_answers = [
        (choice, mmh3.hash(choice), 1) for choice, label in zip(choices, labels) if label == 1
    ]
    incorrect_answers = [
        (choice, mmh3.hash(choice), 0) for choice, label in zip(choices, labels) if label == 0
    ]

    # deterministically sample correct answers if more than max_choices
    n_max_correct = min(max_choices, len(correct_answers))
    correct_answers.sort(key=lambda x: x[1])
    sampled_correct = correct_answers[:n_max_correct]

    # deterministically sample incorrect answers minding max_choices
    n_max_incorrect = max_choices - n_max_correct
    incorrect_answers.sort(key=lambda x: x[1])
    sampled_incorrect = incorrect_answers[:n_max_incorrect]

    # put together and shuffle by sorting by hash
    final_choices_with_hashes = sampled_correct + sampled_incorrect
    final_choices_with_hashes.sort(key=lambda x: x[1])
    shuffled_choices = [choice for choice, _, _ in final_choices_with_hashes]
    shuffled_labels = [label for _, _, label in final_choices_with_hashes]

    # Create the final shuffled dictionary
    # currently ignore mc2_targets and only take only first correct
    shuffled_data = {"choices": shuffled_choices, "label": shuffled_labels.index(1)}

    return shuffled_data
