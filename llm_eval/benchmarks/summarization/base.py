"""
Implementation of summarization benchmarks.

The base class handles the default templating, calculating metrics, etc.
"""

import logging
from abc import abstractmethod

from tqdm import tqdm

from llm_eval.benchmarks import metrics
from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.exceptions import EmptyResponseError

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

    def _run_task(self, llm, results_path=None, n_samples=0):
        """Run the MMLU benchmark using the provided LLM."""
        logging.info(f"Running {self.name} in {n_samples} samples")

        if n_samples:
            indices = self._sample_data(n_samples)
            src_sum = list(zip(self.sources, self.summaries))
            data = [src_sum[ind] for ind in indices]
        else:
            data = zip(self.sources, self.summaries)

        prompt_template = PROMPT_TEMPLATES[self.prompt_type][self.language]
        benchmark_results = []

        for source, summary in tqdm(data, desc=f"Running {self.name}"):
            prompt = prompt_template.format(
                DOCUMENT_TYPE=self.document_type, TARGET_LENGTH=self.target_length, DOCUMENT=source
            )

            result = {
                "source": source,
                "summary": summary,
            }

            try:
                llm_response = llm.prompt(prompt)
                if not llm_response:
                    raise EmptyResponseError
                result["response"] = llm_response
            except Exception as e:
                result["response"] = ""
                result["error"] = True
                result["exception"] = str(e)

            benchmark_results.append(result)

        return benchmark_results

    def _calculate_metric(self, results=None):
        """Given results, calculate desired score"""
        logging.info(f"Calculating Summarization Metrics for {self.name}")
        predictions = [entry["response"] if entry["response"] else "" for entry in results]
        references = [entry["summary"] if entry["summary"] else "" for entry in results]

        rouge_score = metrics.rouge(predictions=predictions, references=references)
        bleu_score = metrics.bleu(predictions=predictions, references=references)
        meteor_score = metrics.meteor(predictions=predictions, references=references)
        bert_score = metrics.bertscore(
            predictions=predictions, references=references, lang=self.language.lower()
        )

        return {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "meteor": meteor_score,
            "bert_score": bert_score,
        }

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "data_path": self.data_path,
            "language": self.language,
            "prompt_type": self.prompt_type,
            "target_length": self.target_length,
            "document_type": self.document_type,
            "prompt_template": PROMPT_TEMPLATES[self.prompt_type][self.language],
            "translator": self.translator.get_metadata() if self.translator else None,
        }
        return metadata
