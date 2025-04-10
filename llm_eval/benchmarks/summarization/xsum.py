"""
Benchmarking (Extreme) Summarization using the XSum dataset which
aims to create a short, one-sentence summary of online
articles from the British Broadcasting Corporation (BBC) [1].

In our experiments we use the test split of the dataset available
in the HuggingFace Hub (https://huggingface.co/datasets/EdinburghNLP/xsum).

We instruct the model to summarize to "1 sentence" according to the task.

References:
[1] Narayan, Shashi, Shay B. Cohen, and Mirella Lapata.
"Don't give me the details, just the summary!
Topic-aware convolutional neural networks for extreme summarization."
arXiv preprint arXiv:1808.08745 (2018).
"""
from datasets import load_dataset

from llm_eval.benchmarks.summarization.huggingface import (
    HuggingFaceSummarizationBaseBenchmark,
)


class XSum(HuggingFaceSummarizationBaseBenchmark):
    """
    The Extreme Summarization.

    By default the benchmark is in English, but if a translator is provided,
    it could be translated and run it other languages.

    An example entry
    (Source: https://huggingface.co/datasets/EdinburghNLP/xsum/viewer/default/train?row=95)
    {
        "Document": (
            "A video was released via social "media from MotoGP's Valencia Grand Prix, appearing "
            "to show the Italian colliding with the fan while riding a motorcycle. The nine-time "
            "world champion apologised for the incident and said that he hoped she was ok. "
            "Rossi, 37, added it was difficult for him to move quickly around the paddock. "
            "Fan Ana Cabanillas Vazquez told Spanish radio station COPE she would have accepted "
            "the apology if she thought it 'had been an accident'."
            "'Seeing the video, you can tell that it was done on purpose,' she said. "
            "I have a small bruise on my leg. I'll consider pressing charges. "
            "Rossi finished fourth in Valencia, the final race of the MotoGP season and "
            "came second in the championship standings behind Spain's Marc Marquez."
        ),
        "Summary": (
            "A fan has threatened to press charges against Valentino Rossi following"
            "an incident in the paddock that occurred while she was taking a selfie."
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
        """Initialize XSum benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            source_field="document",
            summary_field="summary",
            hf_repository="EdinburghNLP/xsum",
            data_dir=data_dir,
            language=language,
            prompt_type=prompt_type,
            target_length=(1, "sentence"),
            document_type="news article",
            translator=translator,
            max_translation_entries=max_translation_entries,
        )

    def _load_huggingface_data(self):
        dataset = load_dataset(self.hf_repository, trust_remote_code=True, split="test")
        return dataset
