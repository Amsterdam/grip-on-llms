"""
Standard vLLM profiles optimized for different model sizes on H100-80GB
Covers 1B to ~80B parameters with optimal settings for each tier
"""

from typing import Any, Dict, Optional


class H100ModelProfiles:
    """Optimized vLLM configurations for different model sizes on H100-80GB"""

    PROFILES = {
        "tiny": {
            "param_range": (1e9, 3e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 128,
                "max_num_batched_tokens": 65536,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",
                "gpu_memory_utilization": 0.80,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 1310172,
                "block_size": 32,
                "enable_prefix_caching": True,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
        },
        "small": {
            "param_range": (3e9, 8e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 64,
                "max_num_batched_tokens": 49152,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 96,
                "max_num_batched_tokens": 65536,
                "block_size": 32,
                "enforce_eager": False,
                "enable_prefix_caching": True,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
        },
        "medium": {
            "param_range": (8e9, 20e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 32768,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
            "quantized": {
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.88,
                "max_num_seqs": 48,
                "max_num_batched_tokens": 49152,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
        },
        "large": {
            "param_range": (20e9, 40e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 24576,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
            "quantized": {
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 24,
                "max_num_batched_tokens": 32768,
                "block_size": 32,
                "enable_prefix_caching": True,
                "max_model_len": 2048,
                "kv_cache_dtype": "auto",
            },
        },
        "xlarge": {
            "param_range": (40e9, 80e9),
            "full_precision": {
                # Most 70B+ models won't fit in FP16/BF16 on 80GB
                "dtype": "auto",
                "gpu_memory_utilization": 0.98,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 16384,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "tensor_parallel_size": 2,  # NEED 2 GPU's
                "kv_cache_dtype": "auto",
            },
            "quantized": {
                "dtype": "half",
                "quantization": "awq",
                "kv_cache_dtype": "auto",
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 12,
                "max_num_batched_tokens": 20480,
                "block_size": 32,
                "enable_prefix_caching": True,
                "max_model_len": 2048,
                "enforce_eager": False,
            },
        },
    }

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
            }
        )

        param_count = sum(profile["param_range"]) / 2  # Estimate middle of range
        profile_name = profile_override
    else:
        config, profile_name, param_count = H100ModelProfiles.get_profile(model_name)

    return config
