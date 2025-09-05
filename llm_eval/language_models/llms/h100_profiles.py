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
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 512,
                "max_num_batched_tokens": 131072,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
            "quantized": {
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.96,
                "max_num_seqs": 1024,
                "max_num_batched_tokens": 262144,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
        },
        "small": {
            "param_range": (3e9, 8e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.94,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 65536,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
            "quantized": {
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 768,
                "max_num_batched_tokens": 196608,
                "block_size": 16,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
        },
        "medium": {
            "param_range": (8e9, 20e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 128,
                "max_num_batched_tokens": 32768,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
            "quantized": {
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.94,
                "max_num_seqs": 384,
                "max_num_batched_tokens": 98304,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
        },
        "large": {
            "param_range": (20e9, 40e9),
            "full_precision": {
                "dtype": "auto",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
            },
            "quantized": {
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.93,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 65536,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
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
                "dtype": "auto",
                "quantization": "awq",
                "gpu_memory_utilization": 0.92,
                "max_num_seqs": 192,
                "max_num_batched_tokens": 49152,
                "block_size": 32,
                "enable_prefix_caching": True,
                "enforce_eager": False,
                "max_model_len": 2048,
                "kv_cache_dtype": "fp8",
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
def get_optimal_vllm_config(model_name: str, profile_override: str = "small") -> Dict[str, Any]:
    """
    Get optimal vLLM configuration for any model size

    Args:
        model_name: HuggingFace model name or path
        profile_override: Override profile category (tiny/small/medium/large/xlarge)

    Returns:
        Dictionary of vLLM engine arguments
    """
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

    profile_name = profile_override

    print(f"Using H100 profile '{profile_name}' for {model_name}")

    return config
