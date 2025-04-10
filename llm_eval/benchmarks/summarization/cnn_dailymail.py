"""
Benchmarking Summarization using a dataset of CNN and Daily Mail articles and their summaries.

The dataset was originally compiled by collecting the original articles together with
suplementing material (provided by the publisher) containing an abstractive summary
(a number of bullet points) covering key aspects of the original article [1].

In our experiments, we use a non-anonymized version of dataset [2],
and more specifically the test split of the "3.0.0" version of the dataset available
in the HuggingFace Hub (https://huggingface.co/datasets/abisee/cnn_dailymail).

We choose to instruct the model to summarize to "3-4 sentences" reflecting
the average number of sentences in the ground truth summaries.

References:
[1] Hermann, Karl Moritz, et al.
"Teaching machines to read and comprehend."
Advances in neural information processing systems 28 (2015).

[2] See, Abigail, Peter J. Liu, and Christopher D. Manning.
"Get to the point: Summarization with pointer-generator networks."
arXiv preprint arXiv:1704.04368 (2017).
"""
from datasets import load_dataset

from llm_eval.benchmarks.summarization.huggingface import (
    HuggingFaceSummarizationBaseBenchmark,
)


class CNNDailyMail(HuggingFaceSummarizationBaseBenchmark):
    """
    CNN / DailyMail Summarization

    By default the benchmark is in English, but if a translator is provided,
    it could be translated and run it other languages.

    An example entry
    (Source: https://huggingface.co/datasets/abisee/cnn_dailymail/viewer/3.0.0/train?&row=192)
    {
        "article": (
            "(CNN) -- A car bomb attack in Algeria has killed three people and wounded 23, "
            "the Algerian Press Service reported. An Algerian policeman stands in front of "
            "destroyed buildings in Thenia. The attack occurred Tuesday near an office housing "
            "judicial police in the city of Thenia, about 50 km (31 miles) east of the capital "
            of Algiers, the agency said. The blast destroyed about 20 houses, and a commission "
            "has been appointed to look after the victims, the press agency said. "
            "Islamic extremists in Algeria and other North African countries have struck several "
            "times in recent years. An al Qaeda affiliate claimed responsibility last year for "
            "the deadliest attack in Algiers in 10 years, a bombing that destroyed the prime "
            "minister's headquarters and a police base, killing at least 24 people and wounding "
            "more than 220. Al Qaeda also took responsibility for a January 2 bombing that killed "
            "four and wounded 20 at a building housing security forces in Naciria, a city about "
            "50 km (31 miles) east of Algiers. E-mail to a friend ."
        ),
        "highlights": (
            "Attack occurred near an office housing judicial police in the city of Thenia . "
            "The blast destroyed about 20 houses; 23 also injured . "
            "Al Qaeda also took responsibility for a January 2 bombing that killed four ."
        )
    }
    """

    def __init__(
        self,
        benchmark_name,
        language="NL",
        prompt_type="simple",
        data_dir=None,
        translator=None,
        max_translation_entries=100,
    ):
        """Initialize CNN/Daily Mail benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            source_field="article",
            summary_field="highlights",
            hf_repository="abisee/cnn_dailymail",
            data_dir=data_dir,
            language=language,
            prompt_type=prompt_type,
            target_length=("3-4", "sentences"),
            document_type="news article",
            translator=translator,
            max_translation_entries=max_translation_entries,
        )

    def _load_huggingface_data(self):
        dataset = load_dataset(self.hf_repository, "3.0.0", split="test")
        return dataset
