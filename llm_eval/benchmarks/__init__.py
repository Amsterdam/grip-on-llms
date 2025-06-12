"""Initialization imports for benchmarks."""

from .arc import ARC
from .mmlu import MMLU
from .simplification import AmsterdamSimplification, INTDuidelijkeTaal
from .summarization import CNNDailyMail, XSum
from .tiny_benchmarks import TinyARC, TinyMMLU, TinyTruthfulQA

__all__ = [
    "MMLU",
    "ARC",
    "INTDuidelijkeTaal",
    "AmsterdamSimplification",
    "XSum",
    "CNNDailyMail",
    "TinyARC",
    "TinyMMLU",
    "TinyTruthfulQA",
]
