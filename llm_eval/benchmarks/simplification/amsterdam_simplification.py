"""
Benchmarking Dutch Text Simplification using a dataset containing
1311 automatically aligned complex-simple sentence pairs stemming
from ~50 documents provided by the Communications Department of
the City of Amsterdam [1].

References:
[1] Vlantis, Daniel, Iva Gornishka, and Shuai Wang.
"Benchmarking the simplification of Dutch municipal text."
Proceedings of the 2024 Joint International Conference on
Computational Linguistics, Language Resources and Evaluation
(LREC-COLING 2024). 2024..
"""
import pandas as pd

from llm_eval.benchmarks.simplification.base import SimplificationBaseBenchmark


class AmsterdamSimplification(SimplificationBaseBenchmark):
    """
    The Amsterdam Simplification dataset is available as a csv file containing
    pairs of Complex & Simple sentences

    An example entry:
    {
        "Complex": (
            "Om overzicht te bieden binnen het ruime scala aan termen
            " binnen de diversiteit, zetten we hier de meest "
            "voorkomende definities op een rij."
        ),
        "Simple": (
            "We gebruiken veel termen binnen de diversiteit. "
            "Voor het overzicht zetten we hier de meest "
            "voorkomende termen en hun betekenis op een rij."
        )
    }
    """

    def __init__(self, benchmark_name, data_path, prompt_type):
        """Initialize AmsterdamSimplification benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            data_path=data_path,
            language="NL",
            prompt_type=prompt_type,
            level="B1",
            granularity="sentence",
        )

        self._load_data()

    def _load_data(self):
        self.data = pd.read_csv(self.data_path)

    def get_sources(self):
        """Get source sentences (complex ones)"""
        return self.data["Complex"].tolist()

    def get_targets(self):
        """Get target sentences (ground-truth)"""
        return self.data["Simple"].tolist()
