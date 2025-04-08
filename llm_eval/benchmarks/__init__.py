"""Initialization imports for benchmarks."""

from .arc import ARC
from .mmlu import MMLU
from .simplification import AmsterdamSimplification, INTDuidelijkeTaal
from .summarization import XSum

__all__ = ["MMLU", "ARC", "INTDuidelijkeTaal", "AmsterdamSimplification", "XSum"]
