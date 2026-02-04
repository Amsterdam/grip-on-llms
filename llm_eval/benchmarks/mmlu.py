"""
Implementation of the MMLU benchmark which contains 57 tasks and
aims to measure world knowledge and problem solving ability [1]

There are many different ways of implementing MMLU
(see https://huggingface.co/blog/open-llm-leaderboard-mmlu).

We choose for a pagmatic user-centered approach which allows us
to compare diverse models (including closed source models)
in a setup relevant to the use of models in a municipal context.

Thus we choose to:
1. generate an answer and compare to the correct answer as opposed to
comparing the corresponding probabilities more closely mimicing direct
user interaction with the model.
2. perform a single pass
3. employ a zero-shot setup as opposed to the commonly used 5-shot setup

References:
[1] Hendrycks, Dan, et al. "Measuring massive multitask language understanding."
arXiv preprint arXiv:2009.03300 (2020).
"""
import json
from collections import Counter
from typing import Dict, List

import requests

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.schemas import BenchmarkEvaluation, RunItem

prompt_template = (
    # "The following is a multiple choice question about {question_type}.\n"
    # "Only answer A, B, C or D.\n"
    "Hier volgt een meerkeuzevraag over {question_type}.\n"
    "Antwoord alleen met de letter A, B, C of D.\n"
    "{instruction}\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n"
    # "Answer:"
    "Antwoord:"
)


class MMLU(BaseBenchmark):
    """
    The MMLU benchmark currently expects a source json file
    with tasks such as:
    {
        "instruction": "Welke van de volgende wordt beschouwd als een zuuranhydride?",
        "option_a": "HCl",
        "option_b": "H2SO3",
        "option_c": "SO2",
        "option_d": "Al(NO3)3",
        "answer": "C",
        "id": "high_school_chemistry/dev/0"
    }

    """

    def __init__(
        self,
        benchmark_name,
        source_url=None,
        data_path=None,
        categories=None,
        preferred_response_format="multiple_choice",
    ):
        """Initialize MMLU benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            source_url=source_url,
            data_path=data_path,
            preferred_response_format=preferred_response_format,
        )

        self.categories = categories
        self.data = None
        self.results = {}

        self._prep_data()

    def _prep_data(self):
        """Download the benchmark data if not available and load it."""
        if not self.data_path.exists():
            self._download_data()
        self._load_data()

    def _download_data(self):
        """Download the data"""
        response = requests.get(self.source_url)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "wb") as f:
            f.write(response.content)

    def _load_data(self):
        self.data = json.load(open(self.data_path, "rb"))
        if self.categories:
            self.data = [
                entry for entry in self.data if any(cat in entry["id"] for cat in self.categories)
            ]

    def _get_hashing_data_for_sampling(self):
        return [entry["id"] for entry in self.data]

    def _run_task(self, llm, system_prompt=None, n_samples=0):
        """Run the MMLU benchmark using the provided LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is not loaded.")

        if n_samples:
            indices = self._sample_data(n_samples)
            data = [self.data[ind] for ind in indices]
        else:
            data = self.data

        prompts = [
            prompt_template.format(
                question_type=entry["id"].split("/")[0].replace("/", " "),
                instruction=entry["instruction"],
                option_a=entry["option_a"],
                option_b=entry["option_b"],
                option_c=entry["option_c"],
                option_d=entry["option_d"],
            )
            for entry in data
        ]
        responses = llm.process_batch(
            prompts, system=system_prompt, response_format=self.preferred_response_format
        )

        run_items = []
        for i, (entry, response) in enumerate(zip(data, responses)):
            expected_answer = entry["answer"]
            is_correct = (
                response.processed_response.strip().lower() == expected_answer.strip().lower()
            )

            run_item = RunItem(
                # LLMResponse fields
                **response.model_dump(),
                # RunItem-specific fields
                prompt=prompts[i],
                prompt_idx_original=i,
                target=expected_answer,
                correct=is_correct,
            )
            run_items.append(run_item)

        return run_items

    def _calculate_metrics(self, run_output: list[RunItem]) -> BenchmarkEvaluation:
        """Given results, calculate desired score."""
        n_correct = sum(1 for entry in run_output if entry.correct)
        accuracy = n_correct / len(run_output) if run_output else 0

        return BenchmarkEvaluation(
            metrics={"acc": accuracy},
            total_samples=len(run_output),
        )

    def _check_validity(self, run_output: List[RunItem], scores: BenchmarkEvaluation) -> Dict:
        """Check the validity of run output and scores"""
        unparseable = [
            entry
            for entry in run_output
            if not entry.processed_response or entry.processed_response not in ["A", "B", "C", "D"]
        ]
        answers = [entry.processed_response for entry in run_output if entry.processed_response]
        answers_counts = Counter(answers)
        most_common_answer_rate = (
            answers_counts.most_common(1)[0][1] / len(answers) if answers else 0
        )

        validity = {
            "n_unparsable_responses": len(unparseable),
            "unparsable_responses_rate": len(unparseable) / len(run_output),
            "answers": answers_counts,
            "most_common_answer_rate": most_common_answer_rate,
            "is_invalid_reasons": [],
        }

        return validity

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "categories": self.categories,
        }
        return metadata
