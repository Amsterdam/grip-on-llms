"""
Implementation of simplification benchmarks.

The base class handles the default templ
"""

from abc import abstractmethod

from tqdm import tqdm

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.benchmarks.simplification import metrics

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
    ):
        """Initialize the benchmark."""
        super().__init__(benchmark_name, data_path=data_path)

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

    def _run_task(self, llm, results_path=None, limit_entries=10):
        """Run the MMLU benchmark using the provided LLM."""
        # Determine the number of entries to process
        if limit_entries > 0:
            sources_to_process = self.sources[:limit_entries]
            targets_to_process = self.targets[:limit_entries]
        else:
            sources_to_process = self.sources
            targets_to_process = self.targets

        prompt_template = PROMPT_TEMPLATES[self.prompt_type][self.language]
        benchmark_results = []

        for source, target in tqdm(
            zip(sources_to_process, targets_to_process), desc=f"Running {self.name}"
        ):
            prompt = prompt_template.format(
                GRANULARITY=self.granularity, LEVEL=self.level, TEXT=source
            )

            llm_response = llm.prompt(prompt)
            result = {
                "source": source,
                "target": target,
                "response": llm_response,
            }
            benchmark_results.append(result)

        return benchmark_results

    def _calculate_metric(self, results=None):
        """Given results, calculate desired score"""
        predictions = [entry["response"] for entry in results]
        n_predictions = len(predictions)
        sources = self.sources[:n_predictions]
        references = self.targets[:n_predictions]

        sari_score = metrics.sari(sources=sources, predictions=predictions, references=references)
        bleu_score = metrics.bleu(predictions=predictions, references=references)
        meteor_score = metrics.meteor(predictions=predictions, references=references)
        bert_score = metrics.bertscore(
            predictions=predictions, references=references, lang=self.language.lower()
        )

        return {
            "bleu": bleu_score,
            "sari": sari_score,
            "meteor": meteor_score,
            "bert_score": bert_score,
        }

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "data_path": self.data_path,
            "language": self.language,
            "prompt_type": self.prompt_type,
            "level": self.level,
            "granularity": self.granularity,
            "prompt_template": PROMPT_TEMPLATES[self.prompt_type][self.language],
        }
        return metadata
