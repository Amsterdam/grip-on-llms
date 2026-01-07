"""
Implementation of summarization benchmarks.

The base class handles the default templating, calculating metrics, etc.
"""

import logging
from abc import abstractmethod
from typing import Dict, List

from llm_eval.benchmarks import metrics
from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.schemas import BenchmarkEvaluation, RunItem

PROMPT_TEMPLATES = {
    "simple": {
        "EN": (
            "Below is a {DOCUMENT_TYPE}.\n"
            "Summarize the document in roughly {TARGET_LENGTH}.\n"
            "Document: {DOCUMENT}\n"
            "Summary:"
        ),
        "NL": (
            "Hier volgt een {DOCUMENT_TYPE}.\n"
            "Vat het document samen in ongeveer {TARGET_LENGTH}.\n"
            "Document: {DOCUMENT}\n"
            "Samenvatting:"
        ),
    },
    "detailed": {
        "EN": (
            "Below is a {DOCUMENT_TYPE}.\n"
            "Summarize the document in roughly {TARGET_LENGTH}, "
            "focusing on the main points."
            "Ensure accuracy and preserve facts, dates, names, etc unaltered.\n"
            "Avoid unnecessary details or opinions.\n"
            "Use clear and concise language, and maintain the tone of voice.\n"
            "Document: {DOCUMENT}\n"
            "Summary:"
        ),
        "NL": (
            "Hier volgt een {DOCUMENT_TYPE}.\n"
            "Vat het document samen in ongeveer {TARGET_LENGTH}, "
            "met de nadruk op de belangrijkste punten."
            "Wees nauwkeurig en behoud feiten, data, namen, etc.\n"
            "Vermijd onnodige details of meningen.\n"
            "Gebruik duidelijke en bondige taal en behoud de stijl van de tekst.\n"
            "Document: {DOCUMENT}\n"
            "Samenvatting:"
        ),
    },
}

granularity_translations = {
    "NL": {
        "words": "woorden",
        "sentence": "zin",
        "sentences": "zinnen",
        "paragraph": "paragrafen",
    },
}


document_type_translations = {
    "NL": {
        "news article": "nieuws artikel",
        "document": "document",
    },
}


class SummarizationBaseBenchmark(BaseBenchmark):
    """
    Summarization benchmarks expect a parallel corpus of full and summarized texts.
    These can be articles, documents, etc.
    Individual benchmarks could possibly contain multiple reference examples.
    """

    def __init__(
        self,
        benchmark_name,
        source_url=None,
        data_dir=None,
        data_path=None,
        hf_repository=None,
        language="NL",
        prompt_type="simple",
        target_length=(50, "words"),
        document_type="document",
        translator=None,
    ):
        """Initialize the benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            source_url=source_url,
            data_dir=data_dir,
            data_path=data_path,
            hf_repository=hf_repository,
        )

        self.language = language
        self.prompt_type = prompt_type
        self.document_type = document_type

        if language == "EN":
            self.target_length = f"{target_length[0]} {target_length[1]}"
            self.document_type = document_type
        else:
            granularity = granularity_translations[language][target_length[1]]
            self.target_length = f"{target_length[0]} {granularity}"
            self.document_type = document_type_translations[language][document_type]

        self.translator = translator

        self._load_data()
        self._sources = self.get_sources()
        self._summaries = self.get_summaries()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError("Implement data loading function")

    @abstractmethod
    def get_sources(self):
        """Get source sentences (complex ones)"""
        raise NotImplementedError("Implement getting sources function")

    @abstractmethod
    def get_summaries(self):
        """Get summaries (ground-truth)"""
        raise NotImplementedError("Implement getting summaries function")

    @property
    def sources(self):
        """Access source text (full document)"""
        return self._sources

    @property
    def summaries(self):
        """Access target summary (ground-truth)"""
        return self._summaries

    def _get_hashing_data_for_sampling(self):
        return [f"{source}-{summary}" for source, summary in zip(self.sources, self.summaries)]

    def _run_task(self, llm, n_samples=0):
        """Run the summarization benchmark using the provided LLM."""
        logging.info(f"Running {self.name} in {n_samples} samples")

        if n_samples:
            indices = self._sample_data(n_samples)
            src_sum = list(zip(self.sources, self.summaries))
            data = [src_sum[ind] for ind in indices]
        else:
            data = list(zip(self.sources, self.summaries))

        # filter samples with issues in either the source or the summary
        data = [(source, summary) for (source, summary) in data if source and summary]

        prompt_template = PROMPT_TEMPLATES[self.prompt_type][self.language]

        prompts = [
            prompt_template.format(
                DOCUMENT_TYPE=self.document_type,
                TARGET_LENGTH=self.target_length,
                DOCUMENT=source,
            )
            for (source, _) in data
        ]
        responses = llm.process_batch(prompts)

        run_items = []
        for i, ((source, summary), response) in enumerate(zip(data, responses)):
            run_item = RunItem(
                # LLMResponse fields
                **response.model_dump(),
                # RunItem-specific fields
                prompt=prompts[i],
                prompt_idx_original=i,
                source=source,
                target=summary,
            )
            run_items.append(run_item)

        return run_items

    def _calculate_metrics(self, run_output: list[RunItem]) -> BenchmarkEvaluation:
        """Given results, calculate desired score"""
        logging.info(f"Calculating Summarization Metrics for {self.name}")
        predictions = [
            entry.processed_response if entry.processed_response else "" for entry in run_output
        ]
        references = [entry.target if entry.target else "" for entry in run_output]

        rouge_score = metrics.rouge(predictions=predictions, references=references)
        bleu_score = metrics.bleu(predictions=predictions, references=references)
        meteor_score = metrics.meteor(predictions=predictions, references=references)
        bert_score = metrics.bertscore(
            predictions=predictions, references=references, lang=self.language.lower()
        )

        return BenchmarkEvaluation(
            metrics={
                "bleu": bleu_score,
                "rouge": rouge_score,
                "meteor": meteor_score,
                "bert_score": bert_score,
            },
            total_samples=len(run_output),
        )

    def _check_validity(self, run_output: List[RunItem], scores: BenchmarkEvaluation) -> Dict:
        """Check the validity of run output and scores"""
        long_response_to_target_ratio = 2
        weirdly_long = [
            entry
            for entry in run_output
            if len(entry.processed_response) > long_response_to_target_ratio * len(entry.target)
        ]

        short_response_to_target_ratio = 0.5
        weirdly_short = [
            entry
            for entry in run_output
            if len(entry.processed_response) < short_response_to_target_ratio * len(entry.target)
        ]
        identical = [
            entry
            for entry in run_output
            if entry.processed_response.strip() == entry.source.strip()
        ]

        validity = {
            "long_response_to_target_ratio": long_response_to_target_ratio,
            "n_long_responses": len(weirdly_long),
            "long_responses_rate": len(weirdly_long) / len(run_output),
            "short_response_to_target_ratio": short_response_to_target_ratio,
            "n_short_responses": len(weirdly_short),
            "short_responses_rate": len(weirdly_short) / len(run_output),
            "n_identical_responses": len(identical),
            "identical_responses_rate": len(identical) / len(run_output),
            "is_invalid_reasons": [],
        }

        identical_response_rate_threshold = 0.5
        if validity["identical_responses_rate"] > identical_response_rate_threshold:
            validity["is_invalid_reasons"].append(
                f"more than {identical_response_rate_threshold * 100}% identical responses"
            )

        return validity

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "language": self.language,
            "prompt_type": self.prompt_type,
            "target_length": self.target_length,
            "document_type": self.document_type,
            "prompt_template": PROMPT_TEMPLATES[self.prompt_type][self.language],
            "translator": self.translator.get_metadata() if self.translator else None,
        }
        return metadata
