"""
Implementation of simplification benchmarks.

The base class handles the default templ
"""

import logging
from abc import abstractmethod
from typing import Dict, List

from llm_eval.benchmarks import metrics
from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.schemas import BenchmarkEvaluation, RunItem

PROMPT_TEMPLATES = {
    "simple": {
        "EN": ("Simplify the following {GRANULARITY} to {LEVEL} level: {TEXT} Simple version: "),
        "NL": (
            "Vereenvoudig de volgende {GRANULARITY} naar {LEVEL}-niveau: {TEXT}"
            "Eenvoudige versie: "
        ),
    },
    "detailed": {
        "EN": (
            "Simplify the following {GRANULARITY} to {LEVEL} level. "
            "Use clear language, short sentences and simple structures. "
            "Avoid jargon, complex or abstract words. Use active voice. "
            "Use inclusive language, so that everyone feels respected "
            "independent of their background, skin color, gender, sexual "
            "orientation, age, or disability."
            "The {GRANULARITY} is: {TEXT}"
            "Simple version: "
        ),
        "NL": (
            "Vereenvoudig de volgende {GRANULARITY} naar {LEVEL}-niveau."
            "Gebruik duidelijke taal, korte zinnen en eenvoudige structuren. "
            "Vermijd vaktaal, complexe of abstracte woorden. Gebruik de actieve vorm. "
            "Gebruik inclusieve taal, zodat iedereen zich gerespecteerd voelt, "
            "ongeacht hun achtergrond, huidskleur, geslacht, seksuele oriëntatie, "
            "leeftijd of beperking."
            "De {GRANULARITY} is: {TEXT}"
            "Eenvoudige versie: "
        ),
    },
}

granularity_translations = {
    "NL": {
        "sentence": "zin",
        "paragraph": "paragraaf",
        "document": "document",
    },
}


class SimplificationBaseBenchmark(BaseBenchmark):
    """
    Simplification benchmarks expect a parallel corpus
    of complex and simple texts.
    These can be at a sentence, paragraph or document level.
    Individual benchmarks could contain multiple reference examples
    or target different (CEFR) level of the final texts.
    """

    def __init__(
        self,
        benchmark_name,
        data_path,
        language="NL",
        prompt_type="detailed",
        level="B1",
        granularity="sentence",
        preferred_response_format=None,
    ):
        """Initialize the benchmark."""
        super().__init__(
            benchmark_name,
            data_path=data_path,
            preferred_response_format=preferred_response_format,
        )

        self.language = language
        self.prompt_type = prompt_type
        self.level = level

        if language == "EN":
            self.granularity = granularity
        else:
            self.granularity = granularity_translations[language][granularity]

        self._load_data()

        self._sources = self.get_sources()
        self._targets = self.get_targets()

    @abstractmethod
    def _load_data(self):
        raise NotImplementedError("Implement data loading function")

    @abstractmethod
    def get_sources(self):
        """Get source sentences (complex ones)"""
        raise NotImplementedError("Implement getting sources function")

    @abstractmethod
    def get_targets(self):
        """Get target sentences (ground-truth)"""
        raise NotImplementedError("Implement getting targets function")

    @property
    def sources(self):
        """Access source sentences (complex ones)"""
        return self._sources

    @property
    def targets(self):
        """Access target sentences (ground-truth)"""
        return self._targets

    def _get_hashing_data_for_sampling(self):
        return [f"{source}-{target}" for source, target in zip(self.sources, self.targets)]

    def _run_task(self, llm, system_prompt=None, n_samples=0):
        """Run the simplification benchmark using the provided LLM."""
        logging.info(f"Running {self.name} in {n_samples} samples")

        if n_samples:
            indices = self._sample_data(n_samples)
            src_trg = list(zip(self.sources, self.targets))
            data = [src_trg[ind] for ind in indices]
        else:
            data = list(zip(self.sources, self.targets))

        prompt_template = PROMPT_TEMPLATES[self.prompt_type][self.language]

        prompts = [
            prompt_template.format(GRANULARITY=self.granularity, LEVEL=self.level, TEXT=source)
            for source, target in data
        ]
        responses = llm.process_batch(prompts, system=system_prompt)

        run_items = []
        for i, ((source, target), response) in enumerate(zip(data, responses)):
            run_item = RunItem(
                # LLMResponse fields
                **response.model_dump(),
                # RunItem-specific fields
                prompt=prompts[i],
                prompt_idx_original=i,
                source=source,
                target=target,
            )
            run_items.append(run_item)

        return run_items

    def _calculate_metrics(self, run_output: list[RunItem]) -> BenchmarkEvaluation:
        """Given results, calculate desired score"""
        logging.info(f"Calculating Simplification Metrics for {self.name}")
        predictions = [entry.processed_response for entry in run_output]
        sources = [entry.source for entry in run_output]
        references = [entry.target for entry in run_output]

        sari_score = metrics.sari(sources=sources, predictions=predictions, references=references)
        bleu_score = metrics.bleu(predictions=predictions, references=references)
        meteor_score = metrics.meteor(predictions=predictions, references=references)
        bert_score = metrics.bertscore(
            predictions=predictions, references=references, lang=self.language.lower()
        )

        return BenchmarkEvaluation(
            metrics={
                "bleu": bleu_score,
                "sari": sari_score,
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
            "level": self.level,
            "granularity": self.granularity,
            "prompt_template": PROMPT_TEMPLATES[self.prompt_type][self.language],
        }
        return metadata
