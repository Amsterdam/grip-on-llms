"""
Base classes for social bias benchmarks.

This module provides the foundation for evaluating social biases in language models,
particularly focused on Dutch cultural context and municipal governance applications.
"""
from abc import abstractmethod
from typing import Any, Dict, Optional

from llm_eval.benchmarks.base import BaseBenchmark


class SocialBiasBenchmark(BaseBenchmark):
    """
    Base class for social bias benchmarks.

    Extends BaseBenchmark to provide specialized functionality for measuring
    social biases across different demographic dimensions (gender, ethnicity,
    age, socioeconomic status, etc.) in Dutch municipal contexts.
    """

    def __init__(
        self,
        benchmark_name: str,
        source_url: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_path: Optional[str] = None,
        hf_repository: Optional[str] = None,
        preferred_response_format: Optional[str] = None,
        language: str = "nl",
    ):
        """
        Initialize social bias benchmark.

        Args:
            benchmark_name: Name of the benchmark
            bias_dimensions: List of bias dimensions to evaluate (e.g., ['gender', 'ethnicity'])
            source_url: URL to download benchmark data
            data_dir: Directory to store benchmark data
            data_path: Path to existing benchmark data
            hf_repository: HuggingFace repository for the benchmark
            preferred_response_format: Expected response format from models
            language: Language for the benchmark (default: 'nl' for Dutch)
        """
        super().__init__(
            benchmark_name=benchmark_name,
            source_url=source_url,
            data_dir=data_dir,
            data_path=data_path,
            hf_repository=hf_repository,
            preferred_response_format=preferred_response_format,
        )

        self._language = language
        self.bias_dimensions = ["Gender", "Origin"]

    @property
    def language(self) -> str:
        """Get the benchmark language"""
        return self._language

    @abstractmethod
    def _calculate_metric(self, results: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Implement getting targets function")

    def _get_own_metadata(self) -> Dict[str, Any]:
        """Get social bias benchmark-specific metadata"""
        return {
            "bias_dimensions": self.bias_dimensions,
            "language": self.language,
            "benchmark_type": "social_bias",
        }

    def _run_task(self, llm, n_samples=0):
        raise NotImplementedError("Implement getting targets function")
