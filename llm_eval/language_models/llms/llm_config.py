"""
Mapping of model names to full HF model ids with H100-optimized profiles.
Each model is assigned to the appropriate H100 profile category based on parameter count.
"""

MODEL_MAPPING = {
    # SMALL MODELS (3B-8B) - H100 "small" profile
    "falcon3-7b-instruct": {
        "id": "tiiuae/Falcon3-7B-Instruct",
        "h100_profile": "small",  # 7B parameters
        "kwargs": {
            # "system_prompt": (
            #     "You are a helpful friendly assistant Falcon3 from TII, "
            #     "try to follow instructions as much as possible."
            # ),
        },
    },
    "mistral-7b-instruct-v0.3": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "h100_profile": "small",  # 7B parameters
        "kwargs": {},
    },
    # LARGE MODELS (20B-40B) - H100 "large" profile
    "mistral-small-instruct": {
        "id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "h100_profile": "large",  # 24B parameters
        "kwargs": {},
    },
    "mistral-large-instruct": {
        "id": "mistralai/Mistral-Large-Instruct-2411",
        "h100_profile": "xlarge",  # ~123B parameters - needs quantization
        "kwargs": {},
    },
    "mistral-large-instruct-quantized": {
        "id": "TechxGenus/Mistral-Large-Instruct-2407-GPTQ",
        "h100_profile": "xlarge",  # Quantized version of large model
        "kwargs": {},
    },
    # TINY MODELS (1B-3B) - H100 "tiny" profile
    "tiny-llama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "h100_profile": "tiny",  # 1.1B parameters
        "kwargs": {},
    },
    "llama-3.2-3b-instruct": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "h100_profile": "tiny",  # 3B parameters
        "kwargs": {},
    },
    # SMALL MODELS (3B-8B) - H100 "small" profile
    "llama-3.1-8b-instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "h100_profile": "small",  # 8B parameters
        "kwargs": {},
    },
    # XLARGE MODELS (40B-80B) - H100 "xlarge" profile - requires quantization
    "llama-3.3-70b-instruct": {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "h100_profile": "xlarge",  # 70B parameters
        "kwargs": {},
    },
    # TINY MODELS (1B-3B) - H100 "tiny" profile
    "phi-4-mini-instruct": {
        "id": "microsoft/Phi-4-mini-instruct",
        "h100_profile": "tiny",  # ~14B parameters (but very efficient architecture)
        "kwargs": {},
    },
    # SMALL MODELS (3B-8B) - H100 "small" profile
    "olmo-7b-instruct": {
        "id": "allenai/OLMo-2-1124-7B-Instruct",
        "h100_profile": "small",  # 7B parameters
        "kwargs": {},
    },
    # LARGE MODELS (20B-40B) - H100 "large" profile
    "olmo-32b-instruct": {
        "id": "allenai/OLMo-2-0325-32B-Instruct",
        "h100_profile": "large",  # 32B parameters
        "kwargs": {},
    },
    # MEDIUM MODELS (8B-20B) - H100 "medium" profile
    "eurollm-9b-instruct": {
        "id": "utter-project/EuroLLM-9B-Instruct",
        "h100_profile": "medium",  # 9B parameters
        "kwargs": {},
    },
    "eurollm-22b-instruct": {
        "id": "utter-project/EuroLLM-22B-Instruct-Preview",
        "h100_profile": "large",  # 22B parameters
        "kwargs": {},
    },
    # SMALL MODELS (3B-8B) - H100 "small" profile
    "qwen-8b": {
        "id": "Qwen/Qwen3-8B",
        "h100_profile": "small",  # 8B parameters
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    # LARGE MODELS (20B-40B) - H100 "large" profile
    "qwen-32b": {
        "id": "Qwen/Qwen3-32B",
        "h100_profile": "large",  # 32B parameters
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    # MEDIUM MODELS (8B-20B) - H100 "medium" profile
    "gemma-12b-instruct": {
        "id": "google/gemma-3-12b-it",
        "h100_profile": "medium",  # 12B parameters
        "kwargs": {},
    },
    # LARGE MODELS (20B-40B) - H100 "large" profile
    "gemma-27b-instruct": {
        "id": "google/gemma-3-27b-it",
        "h100_profile": "large",  # 27B parameters
        "kwargs": {},
    },
}
