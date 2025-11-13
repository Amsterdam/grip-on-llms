"""
Mutliple TinyBenchmarks can be treated as multiple-choice questions,
e.g. TruthfulQA, MMLU, ARC.
This is the implementation of common functionality such as
templating the question and corresponding A/B/C/D answers,
as well as running the task itself and calculating metrics.
"""
from collections import Counter
from typing import Dict, List

from llm_eval.benchmarks.metrics import tiny_scores
from llm_eval.benchmarks.tiny_benchmarks.base import BaseTinyBenchmark
from llm_eval.utils.schemas import BenchmarkEvaluation, RunItem

ANSWERS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
}

ANSWERS_REVERSE = {v: k for k, v in ANSWERS.items()}

PROMPT_TEMPLATES = {
    "EN": (
        "The following is a multiple choice question.\n"
        "Only answer A, B, C or D.\n"
        "{question}\n"
        "Answer:"
    ),
    "NL": (
        "Hier volgt een meerkeuzevraag.\n"
        "Antwoord alleen met de letter A, B, C of D.\n"
        "{question}\n"
        "Antwoord:"
    ),
}


class BaseTinyMultipleChoiceBenchmark(BaseTinyBenchmark):
    """BaseTinyMCBenchmark."""

    def __init__(
        self,
        benchmark_name,
        input_field,
        target_field,
        benchmark_purpose,
        data_dir=None,
        hf_repository=None,
        language="NL",
        translator=None,
        tiny_task=None,
    ):
        """Initialize TinyMultipleChoice benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            input_field=input_field,
            target_field=target_field,
            benchmark_purpose=benchmark_purpose,
            hf_repository=hf_repository,
            data_dir=data_dir,
            language=language,
            translator=translator,
            max_translation_entries=None,
            preferred_response_format="multiple_choice",
        )

        self.tiny_task = tiny_task

    def _get_inputs(self):
        """Get formatted questions"""
        return self.dataset[self.input_field]

    def _get_targets(self):
        """Get ground-truth answers. Map 1/2/3/4s to A/B/C/D"""
        return [ANSWERS[x] for x in self.dataset[self.target_field]]

    def _run_task(self, llm, n_samples=0):
        """Run the tiny multiple-choice benchmark using the provided LLM."""
        if n_samples:
            indices = self._sample_data(n_samples)
            src_sum = list(zip(self.inputs, self.targets))
            data = [src_sum[ind] for ind in indices]
        else:
            data = list(zip(self.inputs, self.targets))

        prompt_template = PROMPT_TEMPLATES[self.language]

        prompts = [prompt_template.format(question=input) for input, target in data]
        responses = llm.process_batch(
            prompts,
            response_format=self.preferred_response_format,
        )

        run_items = []
        for i, ((_, target), response) in enumerate(zip(data, responses)):
            is_correct = response.processed_response.strip().lower() == target.strip().lower()
            run_item = RunItem(
                # LLMResponse fields
                **response.model_dump(),
                # RunItem-specific fields
                prompt=prompts[i],
                prompt_idx_original=i,
                target=target,
                correct=is_correct,
            )
            run_items.append(run_item)

        return run_items

    def _calculate_metrics(self, run_output: list[RunItem]) -> BenchmarkEvaluation:
        """Given results, calculate desired score"""
        n_correct = sum(1 for entry in run_output if entry.correct)
        accuracy = n_correct / len(run_output) if run_output else 0

        tiny_predictions = [entry.correct for entry in run_output]

        return BenchmarkEvaluation(
            metrics={
                "acc": accuracy,
                "tiny_scores": tiny_scores(tiny_predictions, task=self.tiny_task),
            },
            total_samples=len(run_output),
        )

    def _check_validity(self, run_output: List[RunItem], scores: BenchmarkEvaluation) -> Dict:
        """Check the validity of run output and scores"""
        unparseable = [
            entry
            for entry in run_output
            if not entry.processed_response or entry.processed_response not in ANSWERS.values()
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

        # Quality?!
        # answer_rate_threshold = 0.9
        # if validity["most_common_answer_rate"] > answer_rate_threshold:
        #     validity["is_invalid_reasons"].append(
        #         f"more than {answer_rate_threshold * 100}% same answers"
        #     )

        # Quality?!
        # unparseable_threshold = 0.90
        # if validity["unparsable_responses_rate"] > unparseable_threshold:
        #     validity["is_invalid_reasons"].append(
        #         f"more than {unparseable_threshold * 100}% unparsable responses"
        #     )

        return validity


def template_question(question, choices):
    result = f"{question}\n" + "\n".join(
        f"{ANSWERS[ind]}. {choice}" for ind, choice in enumerate(choices)
    )
    return result
