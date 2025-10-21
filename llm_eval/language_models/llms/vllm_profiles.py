"""
Standard vLLM profiles optimized for different model sizes and gpus
Covers 1B to ~80B parameters with optimal settings for each tier
"""
import logging
from typing import Any, Dict, Optional, Tuple

from transformers import AutoConfig

from llm_eval.utils.metadata import get_device_info


class GPUModelProfiles:
    """Optimized vLLM configurations for different model sizes with GPU-specific overrides"""

    # Base configurations that are common across GPU types
    BASE_PROFILES = {
        "tiny": {
            "param_range": (1e9, 3e9),
            "base": {
                "dtype": "auto",
                "block_size": 16,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "auto",
                "enforce_eager": False,
            },
        },
        "small": {
            "param_range": (3e9, 8e9),
            "base": {
                "dtype": "auto",
                "block_size": 16,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "auto",
                "enforce_eager": False,
            },
        },
        "medium": {
            "param_range": (8e9, 20e9),
            "base": {
                "dtype": "auto",
                "block_size": 32,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "auto",
                "enforce_eager": False,
            },
        },
        "large": {
            "param_range": (20e9, 40e9),
            "base": {
                "dtype": "auto",
                "block_size": 32,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "auto",
                "enforce_eager": False,
            },
        },
        "xlarge": {
            "param_range": (40e9, 80e9),
            "base": {
                "dtype": "auto",
                "block_size": 32,
                "enable_prefix_caching": False,
                "kv_cache_dtype": "auto",
                "enforce_eager": False,
            },
        },
    }

    # T4-specific overrides (16GB VRAM, conservative settings)
    T4_OVERRIDES = {
        "tiny": {
            "full_precision": {
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 16384,
                "enforce_eager": True,  # More stable on T4
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
            },
        },
        "small": {
            "full_precision": {
                "gpu_memory_utilization": 0.80,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 16384,
                "enforce_eager": True,
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
            },
        },
        "medium": {
            "full_precision": {
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 16384,
                "enforce_eager": True,
                "tensor_parallel_size": 2,  # Need multiple T4s
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 16,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
            },
        },
        "large": {
            "full_precision": {
                "gpu_memory_utilization": 0.95,
                "max_num_seqs": 2,
                "max_num_batched_tokens": 16384,
                "enforce_eager": True,
                "tensor_parallel_size": 4,  # Need 4+ T4s
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 8,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
            },
        },
        "xlarge": {
            # Only quantized viable on T4
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                "kv_cache_dtype": "fp8",
                "tensor_parallel_size": 2,
            }
        },
    }

    # H100-specific overrides (80GB VRAM, aggressive settings)
    H100_OVERRIDES = {
        "tiny": {
            "full_precision": {
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 256,
                "max_num_batched_tokens": 65536,
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.90,
                "max_num_seqs": 512,
                "max_num_batched_tokens": 131072,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
                "disable_sliding_window": True,
            },
        },
        "small": {
            "full_precision": {
                "gpu_memory_utilization": 0.80,
                "max_num_seqs": 128,
                "max_num_batched_tokens": 32768,
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.85,
                "max_num_seqs": 384,
                "max_num_batched_tokens": 98304,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
                "disable_sliding_window": True,
            },
        },
        "medium": {
            "full_precision": {
                "gpu_memory_utilization": 0.75,
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
            },
            "quantized": {
                "quantization": "awq",
                "gpu_memory_utilization": 0.80,
                "max_num_seqs": 192,
                "max_num_batched_tokens": 49152,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
                "disable_sliding_window": True,
            },
        },
        "large": {
            "full_precision": {
                "gpu_memory_utilization": 0.70,
                "max_num_seqs": 32,
                "max_num_batched_tokens": 8192,
            },
            "quantized": {
                "quantization": "awq_marlin",
                "gpu_memory_utilization": 0.75,
                "max_num_seqs": 128,
                "max_num_batched_tokens": 32768,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
                "disable_sliding_window": True,
            },
        },
        "xlarge": {
            "full_precision": {
                "gpu_memory_utilization": 0.65,
                "max_num_seqs": 4,
                "max_num_batched_tokens": 8192,
                "tensor_parallel_size": 1,  # Need 2+ GPUs for 70B+
            },
            "quantized": {
                "quantization": "awq_marlin",
                "gpu_memory_utilization": 0.70,
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "enable_prefix_caching": True,
                # "kv_cache_dtype": "fp8",
                "disable_sliding_window": True,
            },
        },
    }

    MAX_LEN_CAPS = {
        "h100": {
            # "large": 4096,
            # "medium": 8192,
            # "small": 12288,
            # "tiny": 16384,
            "xlarge": 2048,
            "large": 2048,
            "medium": 3072,
            "small": 4096,
            "tiny": 4096,
            "default": 2048,
        },
        "t4": {
            "xlarge": 1024,
            "large": 2048,
            "medium": 4096,
            "small": 6144,
            "tiny": 8192,
            "default": 2048,
        },
    }

    @classmethod
    def build_config(cls, gpu_type: str, model_size: str, precision: str) -> Dict[str, Any]:
        """
        Build final configuration by merging base + GPU-specific overrides
        Args:
            gpu_type: "t4" or "h100"
            model_size: Profile size ("tiny", "small", etc.)
            precision: "full_precision" or "quantized"
        Returns:
            Complete configuration dictionary
        """
        # Start with base configuration
        if model_size not in cls.BASE_PROFILES:
            raise ValueError(f"Unknown model_size: {model_size}")

        config = cls.BASE_PROFILES[model_size]["base"].copy()

        # Get GPU-specific overrides
        if gpu_type == "t4":
            overrides = cls.T4_OVERRIDES
        elif gpu_type == "h100":
            overrides = cls.H100_OVERRIDES
        else:
            raise ValueError(f"Unknown GPU type: {gpu_type}")

        # Apply GPU-specific overrides
        if model_size in overrides and precision in overrides[model_size]:
            config.update(overrides[model_size][precision])

        return config

    @classmethod
    def detect_gpu_type(cls) -> Tuple[str, int, int]:
        """Detect GPU type and return (gpu_type, vram_gb, gpu_count)"""
        device_info = get_device_info()
        try:
            gpu_count = device_info["gpu_count"]

            if gpu_count == 0:
                return "cpu", 0, 0

            gpu_name = device_info["gpu"]["device_name"]
            vram_gb = device_info["gpu"]["gpu_memory_total_gb"]

            # Detect GPU type from name (map smaller to t4 config, large to h100)
            gpu_name_lower = gpu_name.lower()
            if "h100" in gpu_name_lower or "a100" in gpu_name_lower:
                return "h100", vram_gb, gpu_count
            elif "t4" in gpu_name_lower or "v100" in gpu_name_lower or "rtx" in gpu_name_lower:
                return "t4", vram_gb, gpu_count
            else:
                # Fallback based on VRAM
                return ("h100" if vram_gb >= 40 else "t4"), vram_gb, gpu_count

        except Exception as e:
            logging.warning(f"Could not detect GPU: {e}")
            return "cpu", 0, 0

    @classmethod
    def get_model_max_length(cls, model_id: str) -> int:
        """Dynamically determine max model length from model configuration"""
        try:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

            # Try to get from config attributes
            max_len = cls._get_max_len_from_config(config, model_id)
            if max_len:
                return max_len

            # If nothing worked -> go for some defaults per architecture
            max_len = cls._get_max_len_from_architecture(model_id)
            return max_len

        except Exception as e:
            logging.error(f"Error loading model config for {model_id}: {e}")
            return 4096

    @classmethod
    def _get_max_len_from_config(cls, config, model_id: str) -> int:
        """Extract max length from model config attributes"""
        # Try different common config attributes for max length
        max_length_attrs = [
            "max_position_embeddings",
            "max_sequence_length",
            "context_length",
            "seq_length",
            "max_seq_len",
            "model_max_length",
            "n_positions",
            "max_length",
        ]

        for attr in max_length_attrs:
            if hasattr(config, attr):
                max_len = getattr(config, attr)
                if max_len and max_len > 0:
                    logging.info(
                        f"Found max_model_len={max_len} from config.{attr} for {model_id}"
                    )
                    return max_len

        # Special handling for sliding window
        if hasattr(config, "sliding_window") and config.sliding_window:
            return config.sliding_window

        return 0

    @classmethod
    def _get_max_len_from_architecture(cls, model_id: str) -> int:
        """Get max length based on model architecture"""
        model_name_lower = model_id.lower()
        architecture_defaults = {
            "mistral": 32768,
            "mixtral": 32768,
            "qwen3": 32768,
            "qwen": 8192,
            "llama-3": 131072,
            "llama": 4096,
            "falcon3": 8192,
            "falcon": 2048,
            "phi-4": 4096,
            "phi": 2048,
            "gemma": 8192,
            "olmo": 4096,
            "euro": 4096,
            "fietje": 2048,
            "tinyllama": 2048,
        }
        for key, max_len in architecture_defaults.items():
            if key in model_name_lower:
                logging.info(f"Assuming max_model_len={max_len} for {model_id}")
                return max_len

        # If all else fails -> default to something sensible
        return 4096

    @classmethod
    def _detect_quantization(cls, model_name_or_path: str, loading_kwargs: Dict) -> Optional[str]:
        """Detect quantization type"""
        if "quantization" in loading_kwargs:
            return loading_kwargs["quantization"]

        name_lower = model_name_or_path.lower()
        if "awq" in name_lower:
            return "awq_marlin"
        elif "gptq" in name_lower:
            return "gptq_marlin"
        elif any(q in name_lower for q in ["4bit", "int4", "q4_"]):
            return "gptq_marlin"
        elif "fp8" in name_lower or "8bit" in name_lower:
            return "fp8"

        return None

    @classmethod
    def generate_optimal_config(
        cls, model_name: str, model_size: str = "small", model_loading_config: Dict = None
    ) -> Dict[str, Any]:
        """
        Get optimal vLLM configuration for any model size
        Args:
            model_name: HuggingFace model name or path
            model_size: model sized used to override profile category
                    (tiny/small/medium/large/xlarge)
        Returns:
            Dictionary of vLLM engine arguments
        """
        model_loading_config = model_loading_config or {}

        # Detect hardware
        gpu_type, vram_gb, gpu_count = cls.detect_gpu_type()
        logging.info(f"Detected: {gpu_count}x {gpu_type.upper()} with {vram_gb}GB VRAM each")

        # Determine precision
        quantization = cls._detect_quantization(model_name, model_loading_config)
        precision = "quantized" if quantization else "full_precision"

        # Handle edge cases
        if gpu_type == "t4":
            # Force quantization for large models on T4
            if model_size in ["large", "xlarge"] and not quantization:
                logging.warning(f"Large model {model_name} on T4 - quantization recommended")
            # xlarge only supports quantized on T4
            if model_size == "xlarge" and precision == "full_precision":
                precision = "quantized"
                logging.info("Switching to quantized for xlarge model on T4")

        # Build configuration using clean system
        try:
            config = cls.build_config(gpu_type, model_size, precision)
        except ValueError as e:
            # Fallback to smaller profile
            fallback_profile = "small"
            logging.warning(f"{e}, falling back to {fallback_profile}")
            config = cls.build_config(gpu_type, fallback_profile, precision)

        # Add dynamic max length
        max_model_len = cls.get_model_max_length(model_name)

        # if max_model_len < 1024:
        #     logging.info(f"Context {max_model_len} too small, forcing to 2048")
        #     max_model_len = 2048

        # Limit context if too large for gpu
        max_len_cap = cls.MAX_LEN_CAPS[gpu_type][model_size]
        if max_model_len > max_len_cap:
            logging.info(
                f"Limiting context from {max_model_len} to {max_len_cap}"
                "for {model_size} on {gpu_type}"
            )
            max_model_len = min(max_model_len, max_len_cap)

        config["max_model_len"] = max_model_len

        # Add some default and required vLLM settings (if not there yet)
        config.update(
            {
                "model": model_name,
                "tokenizer": model_name,
                "tensor_parallel_size": config.get("tensor_parallel_size", 1),
                "pipeline_parallel_size": 1,
                "trust_remote_code": True,
            }
        )

        config.update(model_loading_config)

        logging.info(
            f"Using {gpu_type.upper()} {precision} config for {model_name}"
            f"('{model_size}' profile) with max_model_len={max_model_len} & "
            f"max_num_batched_tokens={config['max_num_batched_tokens']}"
        )

        return config


# Convenience function for quick setup
def get_optimal_vllm_config(
    model_name: str, model_size: str = "small", model_loading_config: Dict = None
) -> Dict[str, Any]:
    """Get optimal vLLM config with automatic GPU detection"""
    return GPUModelProfiles.generate_optimal_config(model_name, model_size, model_loading_config)
