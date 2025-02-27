"""
Module for handling benchmarks and running evaluation.
For every benchmark we should be able to provide an LLM,
generate LLM responses and evaluate them.
"""
from abc import abstractmethod


class BaseBenchmark:
    """Base LLM class"""

    @abstractmethod
    def run(self, LLM):
        """Run benchmark for a certain model"""
        raise NotImplementedError("Implement run function")


