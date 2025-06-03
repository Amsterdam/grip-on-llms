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
    "phi-4-mini-instruct": {
        "id": "microsoft/Phi-4-mini-instruct",
        "kwargs": {},
    },
}
