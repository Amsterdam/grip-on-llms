"""Initialization imports for benchmarks."""

from .arc import ARC
from .honesty import HonestCityBench
from .mmlu import MMLU
from .simplification import AmsterdamSimplification, INTDuidelijkeTaal
from .social_bias import BZKSocialBias, DutchBBQ, DutchCrowSPairs
from .summarization import CNNDailyMail, XSum
from .tiny_benchmarks import TinyARC, TinyMMLU, TinyTruthfulQA
from .use_cases import KOGClassifier

__all__ = [
    "MMLU",
    "ARC",
    "INTDuidelijkeTaal",
    "AmsterdamSimplification",
    "BZKSocialBias",
    "DutchBBQ",
    "DutchCrowsPairs",
    "XSum",
    "CNNDailyMail",
    "TinyARC",
    "TinyMMLU",
    "TinyTruthfulQA",
    "HonestCityBench",
    "KOGClassifier",
]
