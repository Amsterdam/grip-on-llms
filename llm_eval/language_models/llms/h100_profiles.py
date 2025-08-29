"""
Standard vLLM profiles optimized for different model sizes on H100-80GB
Covers 1B to ~80B parameters with optimal settings for each tier

Summary of profiles:

TINY (1B-3B): Optimized for throughput
- Higher batch sizes (8-16 sequences)
- Aggressive memory usage (95-98%)
- Full context length

SMALL (3B-8B): Balanced approach
- Moderate batch sizes (4-8 sequences)
- Good memory usage (90-95%)
- Full context length

MEDIUM (8B-20B): Latency focused
- Lower batch sizes (2-4 sequences)
- Conservative memory (85-90%)
- Full context length

LARGE (20B-40B): Memory conscious
- Single request focus (1-2 sequences)
- Conservative memory (80-85%)
- Limited context for full precision

XLARGE (40B-80B): Quantization required
- Single request only
- Requires AWQ/GPTQ for full precision models
- FP8 KV cache essential
- Larger block sizes for efficiency

Key insight: Beyond 40B parameters, quantization becomes essential
for fitting models on a single H100-80GB.
"""

from typing import Any, Dict, Optional


class H100ModelProfiles:
    """Optimized vLLM configurations for different model sizes on H100-80GB"""

    PROFILES = {
        # Tiny models (1B-3B) - Optimize for throughput
        "tiny": {
            "param_range": (1e9, 3e9),
            "full_precision": {
                "dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.95,  # Aggressive - small models
                "max_num_seqs": 8,  # Higher batch size for throughput
                "max_num_batched_tokens": 16384,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": None,  # Use full context
            },
            "quantized": {
                # Usually not needed for tiny models, but for completeness
                "dtype": "half",
                "quantization": "gptq",  # AWQ also fine
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.98,
                "max_num_seqs": 16,  # Even higher throughput
                "max_num_batched_tokens": 32768,
                "block_size": 16,
                "enable_prefix_caching": True,
            },
        },
        # Small models (3B-8B) - Balance throughput and latency
        "small": {
            "param_range": (3e9, 8e9),
            "full_precision": {
                "dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 4,  # Moderate batch size
                "max_num_batched_tokens": 12288,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": None,
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",  # Slightly better than GPTQ for this size
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 20480,
                "block_size": 16,
                "enable_prefix_caching": True,
            },
        },
        # Medium models (8B-20B) - Focus on single request latency
        "medium": {
            "param_range": (8e9, 20e9),
            "full_precision": {
                "dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 2,  # Lower batch for latency
                "max_num_batched_tokens": 8192,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": None,
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 12288,
                "block_size": 20,  # Slightly larger blocks
                "enable_prefix_caching": True,
            },
        },
        # Large models (20B-40B) - Conservative settings
        "large": {
            "param_range": (20e9, 40e9),
            "full_precision": {
                "dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.80,  # More conservative
                "max_num_seqs": 1,  # Single request focus
                "max_num_batched_tokens": 4096,
                "block_size": 20,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 8192,  # Limit context to save memory
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",  # AWQ generally better for large models
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 2,
                "max_num_batched_tokens": 6144,
                "block_size": 24,
                "enable_prefix_caching": True,
                "max_model_len": 16384,  # More context available
            },
        },
        # Extra Large models (40B-80B) - Memory optimized
        "xlarge": {
            "param_range": (40e9, 80e9),
            "full_precision": {
                # Most 70B+ models won't fit in FP16/BF16 on 80GB
                "dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.75,  # Very conservative
                "max_num_seqs": 1,
                "max_num_batched_tokens": 2048,  # Small batches
                "block_size": 32,  # Larger blocks for efficiency
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 4096,  # Limited context
                "swap_space": 8,  # More swap space
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",  # Critical for fitting large models
                "kv_cache_dtype": "fp8",
                "gpu_memory_utilization": 0.88,  # Can be more aggressive
                "max_num_seqs": 1,
                "max_num_batched_tokens": 4096,
                "block_size": 32,
                "enable_prefix_caching": True,
                "max_model_len": 8192,  # More context possible
                "swap_space": 4,
            },
        },
    }

    @classmethod
    def get_profile(
        cls,
        model_name_or_path: str,
        param_count: Optional[int] = None,
        force_quantized: bool = False,
    ) -> Dict[str, Any]:
        """Get optimal profile for a model based on parameter count"""
        if param_count is None:
            param_count = cls._estimate_params(model_name_or_path)

        # Determine size category
        profile_name = cls._get_size_category(param_count)
        profile = cls.PROFILES[profile_name]

        # Choose quantized vs full precision
        is_quantized = cls._detect_quantization(model_name_or_path) or force_quantized

        if is_quantized and "quantized" in profile:
            config = profile["quantized"].copy()
        else:
            config = profile["full_precision"].copy()

        # Add common settings
        config.update(
            {
                "model": model_name_or_path,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "trust_remote_code": True,
                "disable_log_stats": True,
                "disable_log_requests": True,
            }
        )

        # Detect specific quantization type if not set
        if is_quantized and "quantization" in config and config["quantization"] in ["awq", "gptq"]:
            detected_quant = cls._detect_quantization(model_name_or_path)
            if detected_quant:
                config["quantization"] = detected_quant

        return config, profile_name, param_count

    @classmethod
    def _estimate_params(cls, model_name_or_path: str) -> int:
        """Estimate parameter count from model name or config"""
        name_lower = model_name_or_path.lower()

        # Common patterns in model names
        size_patterns = {
            "1b": 1e9,
            "1.3b": 1.3e9,
            "1.5b": 1.5e9,
            "2b": 2e9,
            "2.7b": 2.7e9,
            "3b": 3e9,
            "3.8b": 3.8e9,
            "6b": 6e9,
            "7b": 7e9,
            "8b": 8e9,
            "9b": 9e9,
            "11b": 11e9,
            "13b": 13e9,
            "14b": 14e9,
            "15b": 15e9,
            "20b": 20e9,
            "22b": 22e9,
            "27b": 27e9,
            "30b": 30e9,
            "34b": 34e9,
            "40b": 40e9,
            "45b": 45e9,
            "65b": 65e9,
            "70b": 70e9,
            "72b": 72e9,
            "80b": 80e9,
        }

        for pattern, params in size_patterns.items():
            if pattern in name_lower:
                return int(params)

        # Fallback: try to load config
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

            # Estimate from architecture
            hidden_size = getattr(config, "hidden_size", 4096)
            num_layers = getattr(config, "num_hidden_layers", 32)
            vocab_size = getattr(config, "vocab_size", 32000)

            # Rough estimation for transformer models
            params = vocab_size * hidden_size * 2  # Embeddings
            params += num_layers * hidden_size * hidden_size * 8  # Attention + FFN

            return int(params)
        except Exception:
            # Ultimate fallback - assume medium model
            return int(13e9)

    @classmethod
    def _get_size_category(cls, param_count: int) -> str:
        """Determine size category from parameter count"""
        for category, config in cls.PROFILES.items():
            min_params, max_params = config["param_range"]
            if min_params <= param_count <= max_params:
                return category

        # Fallback for very large models
        if param_count > 80e9:
            return "xlarge"
        else:
            return "medium"

    @classmethod
    def _detect_quantization(cls, model_name_or_path: str) -> Optional[str]:
        """Detect quantization type from model name"""
        name_lower = model_name_or_path.lower()

        if "awq" in name_lower:
            return "awq"
        elif "gptq" in name_lower:
            return "gptq"
        elif any(q in name_lower for q in ["4bit", "int4", "q4_"]):
            return "gptq"
        elif "fp8" in name_lower or "8bit" in name_lower:
            return "fp8"

        return None


# Convenience function for quick setup
def get_optimal_vllm_config(model_name: str, profile_override: str = None) -> Dict[str, Any]:
    """
    Get optimal vLLM configuration for any model size

    Args:
        model_name: HuggingFace model name or path
        for_benchmark: If True, optimize for single-request latency benchmarking
        profile_override: Override profile category (tiny/small/medium/large/xlarge)

    Returns:
        Dictionary of vLLM engine arguments
    """
    # Try to use profile override first, then auto-detect
    if profile_override and profile_override in H100ModelProfiles.PROFILES:
        profile = H100ModelProfiles.PROFILES[profile_override]
        # Determine if model is quantized
        is_quantized = H100ModelProfiles._detect_quantization(model_name)
        if is_quantized and "quantized" in profile:
            config = profile["quantized"].copy()
        else:
            config = profile["full_precision"].copy()

        # Add common settings
        config.update(
            {
                "model": model_name,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "trust_remote_code": True,
                "disable_log_stats": True,
                "disable_log_requests": True,
            }
        )

        param_count = sum(profile["param_range"]) / 2  # Estimate middle of range
        profile_name = profile_override
    else:
        config, profile_name, param_count = H100ModelProfiles.get_profile(model_name)

    # Add metadata for debugging
    config["_profile_name"] = profile_name
    config["_param_count"] = param_count

    return config
