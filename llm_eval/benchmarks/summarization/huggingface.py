"""Base for benchmarking summarization using a dataset from the Hugging Face Hub"""
import logging
from abc import abstractmethod

from datasets import load_dataset

from llm_eval.benchmarks.summarization.base import SummarizationBaseBenchmark
from llm_eval.utils.exceptions import TranslatorMissingError


class HuggingFaceSummarizationBaseBenchmark(SummarizationBaseBenchmark):
    """
    Base for benchmarking summarization using a dataset from the Hugging Face Hub.
    Handles common things like loading a dataset, translating, dumping locally, etc.
    """

    def __init__(
        self,
        benchmark_name,
        source_field,
        summary_field,
        hf_repository,
        data_dir=None,
        language="NL",
        prompt_type="simple",
        target_length=(50, "words"),
        document_type="document",
        translator=None,
        max_translation_entries=100,
    ):
        """Initialize HuggingFaceSummarizationBaseBenchmark."""
        if not source_field or not summary_field:
            ValueError(
                "Both source_field and summary_field must be provided" "for Hugging Face datasets."
            )

        self.source_field = source_field
        self.summary_field = summary_field
        self.max_translation_entries = max_translation_entries

        super().__init__(
            benchmark_name=benchmark_name,
            data_dir=data_dir,
            hf_repository=hf_repository,
            language=language,
            prompt_type=prompt_type,
            target_length=target_length,
            document_type=document_type,
            translator=translator,
        )

    def _load_data(self):
        if self.language != "EN":
            self._load_translated_data()
        else:
            # self.dataset = load_dataset(self.hf_repository, trust_remote_code=True, split="test")
            self.dataset = self._load_huggingface_data()

    @abstractmethod
    def _load_huggingface_data(self):
        raise NotImplementedError("Implement data loading function")

    def _load_translated_data(self):
        """Load the translated dataset (if available) or translate from scratch"""
        if self.translator is None:
            raise TranslatorMissingError(self.language)

        dataset_version = (
            f"{self.language}-"
            f"{self.translator.model_name}-translated-"
            f"{self.max_translation_entries}-samples"
        )
        # Lovely handling of paths, huggingface
        parquet_file = str(self.data_dir / f"{self.name}-{dataset_version}")

        try:
            self.dataset = load_dataset(
                "parquet", data_files={"train": parquet_file}, split="train"
            )
            logging.info(f"Loaded the {dataset_version} translations successfully")
        except Exception as e:
            logging.info(f"Couldn't load the translated parquet for {dataset_version}: {e}")
            logging.info(f"Translating {self.name} to {self.language}")
            self.dataset = self._load_huggingface_data()
            self.dataset = self.dataset.select(range(self.max_translation_entries))

            def try_to_translate(doc):
                try:
                    return self.translator.translate(doc)
                except Exception:
                    return None

            logging.info(f"Need to translate: {len(self.dataset[self.source_field])}")
            for column in [self.source_field, self.summary_field]:
                logging.info(f"Translating {column} column")
                self.dataset = self.dataset.rename_column(column, f"{column}-EN")
                self.dataset = self.dataset.map(
                    lambda entry, column=column: {
                        f"{column}": try_to_translate(entry[f"{column}-EN"])
                    }
                )
                self.dataset = self.dataset.filter(
                    lambda entry, column=column: entry[f"{column}-EN"] is not None
                )

                # Eventually push to hub and load from hub
                # self.dataset.push_to_hub(
                #     f"GemeenteAmsterdam/xsum", dataset_version, private=True, token=self.HF_token
                # )
                self.dataset.to_parquet(parquet_file)
            logging.info(f"Dataset size after translation: {len(self.dataset[self.source_field])}")

    def get_sources(self):
        """Access source text (full document)"""
        return self.dataset[self.source_field]

    def get_summaries(self):
        """Access target summary (ground-truth)"""
        return self.dataset[self.summary_field]
