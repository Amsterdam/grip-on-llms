"""
Political bias benchmarks module.

This module provides benchmarks for evaluating political bias in language models,
specifically focused on Dutch political context using StemWijzer-style questionnaires.
"""

from .stemwijzer import StemWijzerBenchmark

__all__ = ["StemWijzerBenchmark"]
