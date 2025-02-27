"""Mapping of model names to full HF model ids as well as adding necessary parameters."""


MODEL_MAPPING = {
    "falcon-7b-instruct": {
        "id": "tiiuae/falcon-7b-instruct",
        "kwargs": {},
    },
    "falcon-40b-instruct": {
        "id": "tiiuae/falcon-40b-instruct",
        "kwargs": {},
    },
    "mistral-7b-instruct": {
        "id": "mistralai/Mistral-7B-Instruct-v0.1",
        "kwargs": {"torch_dtype": "auto"},
    },
    "mixtral-7b-instruct": {
        "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "kwargs": {"torch_dtype": "auto"},
    },
    "llama-7b-chat": {
        "id": "meta-llama/Llama-2-7b-chat-hf",
        "kwargs": {},
    },
    "llama-13b-chat": {
        "id": "meta-llama/Llama-2-13b-chat-hf",
        "kwargs": {},
    },
    "llama-70b-chat": {
        "id": "meta-llama/Llama-2-70b-chat-hf",
        "kwargs": {},
    },
    "llama3-8b-instruct": {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "kwargs": {},
    },
    "tiny-llama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "kwargs": {},
    },
}
