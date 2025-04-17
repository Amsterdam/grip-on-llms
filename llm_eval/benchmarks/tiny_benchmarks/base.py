"""
TinyBenchmarks are a set of benchmarks (e.g. MMLU, HellaSwag, Winogrande)
which originally consist of tens of thousands of examples but
have been reduced 100 examples which are supposed to be sufficient
to reliably and efficiently reproduce the original evaluation results [1].

The creators test 3 different strategies - stratified random sampling,
clustering examples based on already evaluated LLMs and
an item response theory (IRT) approach - and show that 100 curated examples
are enough to estimate the LLM performance within ~2% error on average.

Translating to Dutch
--------------------
Currently, we use an LLM to translate the English version of the
curated TinyBenchmarks. In future iterations, we would like to make
use of higher quality translations.

Limitations
-----------
The IRT-based strategy for selecting samples is based on the English
versions of the benchmarks. This strategy depends on the performance
of a number of selected LLMs with known correctness data, as well as
a notion of difficulty for the samples. On one hand, a lot of
information is lost in the translation to Dutch. Furthermore,
performance of LLMs differs in low-resource languages such as Dutch.
This all undermines all assumption behind the IRT approach meaning
that the performance guarantees might not necessarily transfer to Dutch.


References:
[1] Polo, Felipe Maia, et al.
"tinyBenchmarks: evaluating LLMs with fewer examples."
arXiv preprint arXiv:2402.14992 (2024).
"""
import logging
from abc import abstractmethod

from datasets import load_dataset
from tqdm import tqdm

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.exceptions import EmptyResponseError, TranslatorMissingError

TRANSLATE_PROMPT = (
    "Below is a formatted prompt for an LLM benchmark."
    "The purpose of the benchmark is to {BENCHMARK_PURPOSE}.\n"
    "Instructions:\n"
    "Your task is to translate the entry to {TARGET_LANGUAGE} "
    "by fully preserving the meaning, the structure, tone of voice. "
    "For multiple-choice questions or continuation tasks "
    "ensure that the translated sentences are grammatically correct "
    "and make sense in the target language. "
    "Make sure that the benchmark is"
    "The entry is:\n"
    "-----"
    "{ENTRY}"
    "-----"
    "Translation: "
)


class BaseTinyBenchmark(BaseBenchmark):
    """
    We make use of the TinyBenchmarks as published in the HF Hub
    (https://huggingface.co/tinyBenchmarks)
    """

    def __init__(
        self,
        benchmark_name,
        input_field,
        target_field,
        data_dir=None,
        hf_repository=None,
        language="NL",
        translator=None,
        max_translation_entries=10,
    ):
        """Initialize the benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            data_dir=data_dir,
            hf_repository=hf_repository,
        )

        self.input_field = input_field
        self.target_field = target_field
        self.language = language
        self.translator = translator
        self.max_translation_entries = max_translation_entries

        self._load_data()
        self._inputs = self._get_inputs()
        self._targets = self._get_targets()

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
            self.dataset = load_dataset("parquet", data_files={"test": parquet_file}, split="test")
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

            input_field = "input_formatted"

            logging.info(f"Need to translate: {len(self.dataset[input_field])}")
            logging.info(f"Translating '{input_field}' column")
            self.dataset = self.dataset.rename_column(input_field, f"{input_field}-EN")
            self.dataset = self.dataset.map(
                lambda entry, input_field=input_field: {
                    f"{input_field}": try_to_translate(entry[f"{input_field}-EN"])
                }
            )
            self.dataset = self.dataset.filter(
                lambda entry, input_field=input_field: entry[f"{input_field}-EN"] is not None
            )

            # Eventually push to hub and load from hub
            # self.dataset.push_to_hub(
            #     f"GemeenteAmsterdam/xsum", dataset_version, private=True, token=self.HF_token
            # )
            self.dataset.to_parquet(parquet_file)
            logging.info(f"Dataset size after translation: {len(self.dataset[input_field])}")

    @property
    def inputs(self):
        """Access source text (full document)"""
        return self._inputs

    @property
    def targets(self):
        """Access target summary (ground-truth)"""
        return self._targets

    def _get_hashing_data_for_sampling(self):
        return self.inputs

    def _run_task(self, llm, results_path=None, n_samples=0):
        """Run the MMLU benchmark using the provided LLM."""
        if n_samples:
            indices = self._sample_data(n_samples)
            src_sum = list(zip(self.inputs, self.targets))
            data = [src_sum[ind] for ind in indices]
        else:
            data = zip(self.inputs, self.targets)

        benchmark_results = []

        for input, target in tqdm(data, desc=f"Running {self.name}"):
            result = {
                "input": input,
                "target": target,
            }
            try:
                llm_response = llm.prompt(input)
                if not llm_response:
                    raise EmptyResponseError
                result["response"] = llm_response
            except Exception as e:
                result["response"] = "FAILED"
                result["error"] = True
                result["exception"] = str(e)
            benchmark_results.append(result)

        return benchmark_results

    @abstractmethod
    def _calculate_metric(self, results=None):
        raise NotImplementedError("Implement metric calculation")

    def _get_own_metadata(self):
        """Get benchmark metadata for versioning purposes"""
        metadata = {
            "data_path": self.data_path,
            "language": self.language,
            # "prompt_template": PROMPT_TEMPLATES[self.prompt_type][self.language],
        }
        return metadata
