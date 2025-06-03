"""
Implementation of the AI2’s Reasoning Challenge (ARC) benchmark [1].

ARC is a common sense reasoning, multiple choice question-answering
dataset, containing questions from science exams from grade 3 to grade 9.
The dataset is split in two partitions: Easy and Challenge, where the
latter partition contains the more difficult questions that require
reasoning. Most of the questions have 4 answer choices, with <1% of all
the questions having either 3 or 5 answer choices. ARC includes a
supporting KB of 14.3M unstructured text passages.

The dataset we use is a Dutch translation of the original dataset,
and can be obtained here:
http://nlp.uoregon.edu/download/okapi-eval/datasets/m_arc/.

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
[1] Clark, Peter, et al.
"Think you have solved question answering? try arc, the ai2 reasoning challenge."
arXiv preprint arXiv:1803.05457 (2018).
"""

import json

import requests
from tqdm import tqdm

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.exceptions import EmptyResponseError

prompt_template = (
    # "The following is a multiple choice question.\n"
    # "Only answer with the letter A, B, C or D\n"
    "Hier volgt een meerkeuzevraag.\n"
    "Antwoord alleen met de letter A, B, C of D.\n"
    "{instruction}\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "C. {option_c}\n"
    "D. {option_d}\n"
    # "Answer:"
    "Antwoord:"
)


class ARC(BaseBenchmark):
    """Evaluates LLMs using the ARC benchmark.

    The ARC benchmark currently expects a source json file with tasks such as:

    {
        "id": "ARC-Challenge/train/Mercury_SC_413300",
        "answer": "C",
        "instruction": "De temperatuur van het water in een glas verandert
        van 5°C naar -1°C. Hoe zal het water waarschijnlijk veranderen?",
        "option_a": "Het zal koken.",
        "option_b": "Het zal smelten.",
        "option_c": "Het zal bevriezen.",
        "option_d": "Het zal condenseren."
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
        """Initialize ARC benchmark."""
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
            print(f"Missing {self.data_path}")
            self._download_data()
        self._load_data()

    def _download_data(self):
        """Download the data."""
        response = requests.get(self.source_url)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "wb") as f:
            f.write(response.content)

    def _load_data(self):
        self.data = json.load(open(self.data_path, "rb"))
        if self.categories:
            # fmt: off
            self.data = [
                entry for entry in self.data
                if any(cat in entry["id"] for cat in self.categories)
            ]
            # fmt: on

    def _get_hashing_data_for_sampling(self):
        return [entry["id"] for entry in self.data]

    def _run_task(self, llm, results_path=None, n_samples=0):
        """Run the ARC benchmark using the provided LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is not loaded.")

        if n_samples:
            indices = self._sample_data(n_samples)
            data = [self.data[ind] for ind in indices]
        else:
            data = self.data

        benchmark_results = []
        for entry in tqdm(data):
            prompt = prompt_template.format(
                instruction=entry["instruction"],
                option_a=entry["option_a"],
                option_b=entry["option_b"],
                option_c=entry["option_c"],
                option_d=entry["option_d"],
            )

            expected_answer = entry["answer"]

            result = {
                "prompt": prompt,
                "expected": expected_answer,
            }

            try:
                llm_response = llm.prompt(prompt, response_format=self.preferred_response_format)
                if not llm_response:
                    raise EmptyResponseError
                result["response"] = llm_response
                result["correct"] = llm_response.strip().lower() == expected_answer.strip().lower()
            except Exception as e:
                result["response"] = ""
                result["error"] = True
                result["exception"] = str(e)
                result["correct"] = False

            benchmark_results.append(result)

        return benchmark_results

    def _calculate_metric(self, results=None):
        """Given results, calculate desired score."""
        accuracy = len([entry for entry in results if entry["correct"]]) / len(results)
        return {"acc": accuracy}

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {"source_url": self.source_url}
        return metadata
