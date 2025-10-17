"""
Mapping of model names to full HF model ids with H100-optimized profiles.
Each model is assigned to the appropriate H100 profile category based on parameter count.
"""

MODEL_MAPPING = {
    # TINY MODELS (1B-3B) - H100 "tiny" profile
    "tiny-llama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_size": "tiny",
        "kwargs": {},
    },
    "llama-3.2-3b-instruct": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "model_size": "tiny",
        "kwargs": {},
    },
    "phi-4-mini-instruct": {
        "id": "microsoft/Phi-4-mini-instruct",
        "model_size": "tiny",
        "kwargs": {},
    },
    "smollm3-3b": {
        "id": "HuggingFaceTB/SmolLM3-3B",
        "model_size": "tiny",
        "kwargs": {},
    },
    "fietje-2-instruct": {
        "id": "BramVanroy/fietje-2-instruct",
        "model_size": "tiny",
        "kwargs": {},
    },

    # SMALL MODELS (3B-8B) - H100 "small" profile
    "falcon3-7b-instruct": {
        "id": "tiiuae/Falcon3-7B-Instruct",
        "model_size": "small",
        "kwargs": {},
    },
    "mistral-7b-instruct-v0.3": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_size": "small",
        "kwargs": {},
    },
    "llama-3.1-8b-instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_size": "small",
        "kwargs": {},
    },
    "olmo-7b-instruct": {
        "id": "allenai/OLMo-2-1124-7B-Instruct",
        "model_size": "small",
        "kwargs": {},
    },
    "qwen-8b": {
        "id": "Qwen/Qwen3-8B",
        "model_size": "small",
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    "command-r7b": {
        "id": "CohereLabs/c4ai-command-r7b-12-2024",
        "model_size": "small",
        "kwargs": {},
    },
    "geitje-7b-ultra": {
        "id": "BramVanroy/GEITje-7B-ultra",
        "model_size": "small",
        "kwargs": {},
    },
    "apertus-8b-instruct": {
        "id": "swiss-ai/Apertus-8B-Instruct-2509",
        "model_size": "small",
        "kwargs": {},
    },

    # MEDIUM MODELS (8B-20B) - H100 "medium" profile
    "eurollm-9b-instruct": {
        "id": "utter-project/EuroLLM-9B-Instruct",
        "model_size": "medium",
        "kwargs": {},
    },
    "gemma-12b-instruct": {
        "id": "google/gemma-3-12b-it",
        "model_size": "medium",
        "kwargs": {},
    },
    "gpt-oss-20b": {
        "id": "openai/gpt-oss-20b",
        "model_size": "medium",
        "kwargs": {},
    },

    # LARGE MODELS (20B-40B) - H100 "large" profile
    "mistral-small-instruct": {
        "id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "model_size": "large",
        "kwargs": {},
    },
    "eurollm-22b-instruct": {
        "id": "utter-project/EuroLLM-22B-Instruct-Preview",
        "model_size": "large",
        "kwargs": {},
    },
    "gemma-27b-instruct": {
        "id": "google/gemma-3-27b-it",
        "model_size": "large",
        "kwargs": {},
    },
    "olmo-32b-instruct": {
        "id": "allenai/OLMo-2-0325-32B-Instruct",
        "model_size": "large",
        "kwargs": {},
    },
    "qwen-32b": {
        "id": "Qwen/Qwen3-32B",
        "model_size": "large",
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    "qwen3-32b-awq": {
        "id": "Qwen/Qwen3-32B-AWQ",
        "model_size": "large",
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    "aya-expanse-32b": {
        "id": "CohereLabs/aya-expanse-32b",
        "model_size": "large",
        "kwargs": {},
    },
    "deepseek-r1-distill-qwen-32b": {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "model_size": "large",
        "kwargs": {},
    },
    # "llama-3.3-70b-instruct": {
    #     "id": "meta-llama/Llama-3.3-70B-Instruct",
    #     "model_size": "xlarge",
    #     "kwargs": {},
    # },
    "llama-3.3-70b-gptq": {
        "id": "shuyuej/Llama-3.3-70B-Instruct-GPTQ",
        "model_size": "xlarge",
        "kwargs": {
            "loading": {
                "quantization": "gptq",
                "dtype": "float16",
            }
        },
    },
    "deepseek-r1-distill-llama-70b-awq": {
        "id": "Valdemardi/DeepSeek-R1-Distill-Llama-70B-AWQ",
        "model_size": "xlarge",
        "kwargs": {
            "loading": {
                "quantization": "awq",
                "dtype": "float16",
            }
        },
    },
    "apertus-70b-instruct-quantized": {
        "id": "RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16",
        "model_size": "large",
        "kwargs": {},
    },
    "gpt-oss-120b": {
        "id": "openai/gpt-oss-120b",
        "model_size": "large",
        "kwargs": {},
    },
    "mistral-large-instruct-2407-awq": {
        "id": "TTechxGenus/Mistral-Large-Instruct-2411-AWQ",
        "model_size": "xlarge",
        "kwargs": {},
    },
}