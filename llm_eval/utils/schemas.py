"""Schemas for LLM responses, benchmark results, validators, etc"""
# flake8: noqa: D106
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class BenchCosts(BaseModel):
    """Cost information for a benchmark run."""

    # Core fields
    total_cost: Optional[float] = Field(None, description="Total cost in euros")
    cost_per_prompt: Optional[float] = Field(None, description="Cost per prompt in euros")
    n_samples: Optional[int] = Field(
        None, description="Number of valid samples used in calculation"
    )
    method: Literal["api", "gpu", ""] = Field("", description="Cost calculation method")
    error: bool = Field(default=False, description="Whether cost calculation failed")

    # API-specific fields
    api_pricing: Optional[dict] = Field(
        None, description="API pricing: {'input': x, 'output': y} per 1k tokens"
    )
    token_counts: Optional[dict] = Field(
        None, description="Token counts: {'input': x, 'output': y}"
    )

    # GPU-specific fields
    gpu_type: Optional[str] = Field(
        None, description="GPU type (e.g., 'Tesla T4', 'NVIDIA H100 NVL')"
    )
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds")
    gpu_hourly_rate: Optional[float] = Field(None, description="GPU hourly rate in euros")

    class Config:
        extra = "allow"


class LLMResponse(BaseModel):
    """Storing the inputs and outputs of LLMs"""

    raw_prompt: str = ""
    formatted_prompt: Union[str, List] = Field(default="")
    raw_response: str = ""
    processed_response: Optional[str] = None
    error: bool = False
    exception: Optional[str] = None

    class Config:
        extra = "allow"


class RunItem(LLMResponse):
    """LLMResponse extended with benchmark-specific fields"""

    # Benchmark entry metadata
    prompt: Optional[str] = None  # May differ from raw_prompt
    prompt_idx_original: Optional[int] = None
    category: Optional[str] = None
    source: Optional[str] = ""

    # Answer evaluation fields
    target: Optional[str] = ""
    correct: Optional[bool] = None

    # Per-item evaluation (e.g., judge scores)
    eval: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class EvaluationMetadata(BaseModel):
    """Metadata about how evaluation was performed"""

    judges: Optional[List[str]] = None
    method: Optional[str] = None  # e.g., "exact_match", "llm_judge", "bert_score"

    class Config:
        extra = "allow"


class LLMMetadata(BaseModel):
    """LLM configuration metadata"""

    model_name: str
    inference_engine: str
    params: Optional[Union[Dict[str, Any], str]] = None  # Dict or string

    class Config:
        extra = "allow"


class BenchmarkMetadata(BaseModel):
    """Benchmark configuration metadata"""

    name: str
    source_url: Optional[str] = None
    data_path: Optional[str] = None
    preferred_response_format: Optional[str] = None
    language: Optional[str] = None
    translation_prompt: Optional[str] = None
    prompt_template: Optional[str] = None

    class Config:
        extra = "allow"


class RunMetadata(BaseModel):
    """Metadata about the benchmark run"""

    timestamp: str
    timestamp_bench_start: Optional[str] = None
    timestamp_bench_end: Optional[str] = None
    time_bench_total: Optional[str] = None
    duration_seconds: Optional[float] = None
    system: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class MetadataContainer(BaseModel):
    """Container for all metadata"""

    llm: LLMMetadata
    benchmark: BenchmarkMetadata
    run: RunMetadata
    code_carbon: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    n_samples: int

    class Config:
        extra = "allow"


class BenchmarkEvaluation(BaseModel):
    """Structured evaluation results"""

    metrics: Dict[str, Any]  # Main metrics: {"acc": 0.71, "f1": 0.68, ...}
    total_samples: int = Field(ge=0)
    eval_metadata: Optional[EvaluationMetadata] = None

    class Config:
        extra = "allow"


class BenchmarkResult(BaseModel):
    """Complete benchmark result with metadata"""

    metadata: MetadataContainer
    run_output: List[Union[RunItem, Dict[str, Any]]]
    evaluation: BenchmarkEvaluation
    costs: Optional[BenchCosts] = None
    validity: Optional[Dict] = None

    # Error tracking
    error: bool = False
    exception: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to JSON format matching your actual structure"""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict) -> "BenchmarkResult":
        """Load from JSON"""
        return cls(**data)

    def save(self, path: Path):
        """Save to JSON file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False, default=str)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResult":
        """Load and validate from JSON file"""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # properties to access nested data
    @property
    def model_name(self) -> str:
        return self.metadata.llm.model_name

    @property
    def benchmark_name(self) -> str:
        return self.metadata.benchmark.name

    @property
    def metrics(self) -> Dict:
        return self.evaluation.metrics

    @property
    def total_samples(self) -> int:
        return self.evaluation.total_samples

    @property
    def n_correct(self) -> int:
        """Count correct answers"""
        return sum(
            1
            for item in self.run_output
            if (item.correct if isinstance(item, RunItem) else item.get("correct", False))
        )

    @property
    def errors_only(self) -> List[Dict]:
        """Get only incorrect items"""
        errors = []
        for item in self.run_output:
            is_correct = item.correct if isinstance(item, RunItem) else item.get("correct", True)
            if not is_correct:
                errors.append(item.model_dump() if isinstance(item, RunItem) else item)
        return errors
