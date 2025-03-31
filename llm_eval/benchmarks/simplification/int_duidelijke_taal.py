"""
Benchmarking Dutch Text Simplification using a dataset containing
sentences from the SoNaR corpus [1] automatically simplified using
GPT-4o and manually evaluated for simplicity, accuracy and fluency [2].

Due to the nature of the simplifications (produced automatically by an LLM),
we use the evaluation scores to filter higher quality data as follows:
- we select samples with "Accuratesse Gem." > 70
- we preserve all samples independent of their Fluency (both for A and B)
- we only preserve samples for which the simplification was scored as simpler
    than the original (that is, "Complexiteit (A) Gem." > "Complexiteit (B) Gem.")

References:
[1] Oostdijk, Nelleke, et al. "SoNaR user documentation."
Online:< https://ticclops. uvt. nl/SoNaR_end-user_documentation_v 1.4 (2013).

[2] Menselijke evaluatie van geautomatiseerde tekstvereenvoudiging:
resultaten van crowdsourcing (Version 1.0) (2024) [Data set].
Available at the Dutch Language Institute: https://hdl.handle.net/10032/tm-a2-y8
"""

import pandas as pd

from llm_eval.benchmarks.simplification.base import SimplificationBaseBenchmark


class INTDuidelijkeTaal(SimplificationBaseBenchmark):
    """
    The INT Duidelijke Taal dataset is available as a csv file containing (among others)
    Niet synthetische tekst/zin (A)
    Synthetische tekst/zin (B)
    Paarsgewijze vergelijking Gem.
    Accuratesse Gem.
    Fluency (A) Gem.
    Fluency (B) Gem.
    Complexiteit (A) Gem.
    Complexiteit (B) Gem.

    An example entry
    {
        "Niet synthetische tekst/zin (A)": (
            "De premier vormt een regering, die vervolgens"
            "door de president officieel wordt benoemd."
        ),
        "Synthetische tekst/zin (B)": (
            "De premier stelt een regering samen en daarna"
            "benoemt de president deze officieel."
        ),
        "Paarsgewijze vergelijking Gem.": 60.625,
        "Accuratesse	Accuratesse Gem.": 98,
        "Fluency (A) Gem.": 75.4,
        "Fluency (B) Gem.": 86.33333333,
        "Complexiteit (A) Gem.": 49.66666667,
        "Complexiteit (B) Gem.": 32.14285714,
    }

    """

    def __init__(self, benchmark_name, data_path, prompt_type):
        """Initialize INT Duidelijke Taal benchmark."""
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
        self.data = self.data[
            (self.data["Accuratesse Gem."] > 70)
            & (self.data["Complexiteit (A) Gem."] > self.data["Complexiteit (B) Gem."])
        ]

    def get_sources(self):
        """Get source sentences (complex ones)"""
        return self.data["Niet synthetische tekst/zin (A)"].tolist()

    def get_targets(self):
        """Get target sentences (ground-truth)"""
        return self.data["Synthetische tekst/zin (B)"].tolist()
