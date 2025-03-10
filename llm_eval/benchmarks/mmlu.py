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
from pathlib import Path

import requests
from tqdm import tqdm

from llm_eval.benchmarks.base import BaseBenchmark

prompt_template = (
    "The following is a multiple choice question about {question_type}.\n"
    "Only answer A, B, C or D.\n"
    "{instruction}\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n"
    "Answer:"
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

    def __init__(self, source_url, data_dir, categories=None):
        self.name = "MMLU-NL"
        self.source_url = source_url
        self.data_dir = Path(data_dir) / self.name
        self.data_path = self.data_dir / "data.json"
        self.categories = categories
        self.data = None
        self.results = {}

        self._prep_data()

    def _prep_data(self):
        """Download the benchmark data if not available and load it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_path.exists():
            self._download_data()
        self._load_data()

    def _download_data(self):
        """Download the data"""
        response = requests.get(self.source)
        with open(self.data_path, "wb") as f:
            f.write(response.content)

    def _load_data(self):
        self.data = json.load(open(self.data_path, "rb"))
        if self.categories:
            self.data = [
                entry for entry in self.data if any(cat in entry["id"] for cat in self.categories)
            ]

    def _run_task(self, llm, results_path=None):
        """Run the MMLU benchmark using the provided LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is not loaded.")

        benchmark_results = []
        for entry in tqdm(self.data):
            question_type = entry["id"].split("/")[0].replace("/", " ")

            prompt = prompt_template.format(
                question_type=question_type,
                instruction=entry["instruction"],
                option_a=entry["option_a"],
                option_b=entry["option_b"],
                option_c=entry["option_c"],
                option_d=entry["option_d"],
            )

            expected_answer = entry["answer"]
            llm_response = llm.prompt(prompt)
            result = {
                "prompt": prompt,
                "expected": expected_answer,
                "response": llm_response,
                "correct": llm_response.strip().lower() == expected_answer.strip().lower(),
            }
            benchmark_results.append(result)

        return benchmark_results

    def _calculate_metric(self, results=None):
        """Given results, calculate desired score"""
        score = len([entry for entry in results if entry["correct"]]) / len(results)
        return score
