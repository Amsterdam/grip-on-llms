"""Mapping of model names to full HF model ids as well as adding necessary parameters."""


MODEL_MAPPING = {
    "falcon3-7b-instruct": {
        "id": "tiiuae/Falcon3-7B-Instruct",
        "kwargs": {
            # "system_prompt": (
            #     "You are a helpful friendly assistant Falcon3 from TII, "
            #     "try to follow instructions as much as possible."
            # ),
        },
    },
    "mistral-7b-instruct-v0.3": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "kwargs": {},
    },
    "mistral-small-instruct": {
        "id": "mistralai/Mistral-Small-24B-Instruct-2501",
        "kwargs": {},
    },
    "mistral-large-instruct": {
        "id": "mistralai/Mistral-Large-Instruct-2411",
        "kwargs": {},
    },
    "tiny-llama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "kwargs": {},
    },
    "llama-3.2-3b-instruct": {
        "id": "meta-llama/Llama-3.2-3B-Instruct",
        "kwargs": {},
    },
    "llama-3.1-8b-instruct": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "kwargs": {},
    },
    "llama-3.3-70b-instruct": {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "kwargs": {},
    },
    "phi-4-mini-instruct": {
        "id": "microsoft/Phi-4-mini-instruct",
        "kwargs": {},
    },
    "olmo-7b-instruct": {
        "id": "allenai/OLMo-2-1124-7B-Instruct",
        "kwargs": {},
    },
    "olmo-32b-instruct": {
        "id": "allenai/OLMo-2-0325-32B-Instruct",
        "kwargs": {},
    },
    "eurollm-9b-instruct": {
        "id": "utter-project/EuroLLM-9B-Instruct",
        "kwargs": {},
    },
    "eurollm-22b-instruct": {
        "id": "utter-project/EuroLLM-22B-Instruct-Preview",
        "kwargs": {},
    },
    "qwen-8b": {
        "id": "Qwen/Qwen3-8B",
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    "qwen-32b": {
        "id": "Qwen/Qwen3-32B",
        "kwargs": {
            "template": {
                "enable_thinking": False,
            }
        },
    },
    "gemma-12b-instruct": {
        "id": "google/gemma-3-12b-it",
        "kwargs": {},
    },
    "gemma-27b-instruct": {
        "id": "google/gemma-3-27b-it",
        "kwargs": {},
    },
}
