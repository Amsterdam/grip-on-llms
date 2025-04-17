"""Initialization imports for benchmarks."""

from .arc import ARC
from .mmlu import MMLU
from .simplification import AmsterdamSimplification, INTDuidelijkeTaal
from .summarization import CNNDailyMail, XSum
from .tiny_benchmarks import TinyMMLU

__all__ = [
    "MMLU",
    "ARC",
    "INTDuidelijkeTaal",
    "AmsterdamSimplification",
    "XSum",
    "CNNDailyMail",
    "TinyMMLU",
]
