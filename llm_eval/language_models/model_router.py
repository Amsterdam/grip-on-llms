"""
Functionality for routing to different LLMs.
Currently supports Azure OpenAI deployments, HuggingFace models, and vLLM inference.
vLLM is now the default provider for local models with H100-optimized profiles.

Usage Examples:
    model = LLMRouter.get_model(model_name="llama-3.1-8b-instruct")  # Uses vLLM by default
    model = LLMRouter.get_model(provider="huggingface", model_name="falcon-7b")  # Explicit HF
    model = LLMRouter.get_model(provider="azure", model_name="gpt-4")  # Azure OpenAI
"""

import logging

from llm_eval.language_models.llms.h100_profiles import get_optimal_vllm_config
from llm_eval.language_models.llms.huggingface import HuggingFaceLLM
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.language_models.llms.openai import OpenAILLM
from llm_eval.language_models.llms.vllm_llm import VLLMLlm


class LLMRouter:
    """Route LLMs depending on model and desired provider."""

    @staticmethod
    def get_model(
        model_name,
        provider=None,  # Now optional - defaults to vLLM for local models
        api_endpoint=None,
        api_key=None,
        api_version=None,
        hf_token=None,
        hf_cache=None,
        params=None,
        uses_api=None,  # Auto-detected based on provider
        # vLLM-specific parameters (ignored for other providers)
        tensor_parallel_size=1,
        gpu_memory_utilization=None,  # Auto-set from H100 profiles
        max_model_len=None,  # Auto-set from H100 profiles
        trust_remote_code=True,  # Default True for vLLM
    ):
        """Get corresponding LLM instance based on model name and optional provider.

        Args:
            model_name (str): The name of the model to load.
            provider (str, optional): The provider of the model.
                Defaults to "vllm" for local models. Supported: "azure", "huggingface", "vllm".
            api_endpoint (str, optional): The endpoint for API-based models (e.g. Azure OpenAI).
            api_key (str, optional): The API key for authentication (for API-based models).
            api_version (str, optional): The API version for API-based models.
            hf_token (str, optional): The Hugging Face token for accessing private models.
            hf_cache (str, optional): Path to the local cache for Hugging Face models.
            params (dict, optional): Additional parameters for the model.
            uses_api (bool, optional): Whether the model uses an API. Auto-detected if None.
            tensor_parallel_size (int, optional): Number of GPUs for tensor parallelism.
            gpu_memory_utilization (float, optional): GPU memory usage
                (auto-set from H100 profiles).
            max_model_len (int, optional): Maximum model sequence length
                (auto-set from profiles).
            trust_remote_code (bool, optional): Whether to trust remote code.

        Returns:
            An instance of `OpenAILLM`, `HuggingFaceLLM`, or `VLLMLlm`.

        Raises:
            NotImplementedError: If an unsupported model is requested on Azure.
            ValueError: If an unknown provider is specified.
        """
        logging.info(f"Getting a model. Provider: {provider}; Model: {model_name}")

        if provider == "azure":
            if "gpt" in model_name:
                return OpenAILLM(
                    model_name=model_name,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    params=params,
                    uses_api=uses_api,
                )
            else:
                raise NotImplementedError(
                    "Currently there is no support for models other than GPT on Azure."
                )
        elif provider == "huggingface":
            return HuggingFaceLLM(
                model_name=model_name,
                hf_token=hf_token,
                hf_cache=hf_cache,
                params=params,
                uses_api=uses_api,
            )
        elif provider == "vllm":
            model_config = MODEL_MAPPING.get(model_name, {})
            model_id = model_config.get("id", model_name)
            profile_override = model_config.get("h100_profile")

            vllm_config = get_optimal_vllm_config(model_id, profile_override=profile_override)

            # Override with user-specified parameters
            if gpu_memory_utilization is not None:
                vllm_config["gpu_memory_utilization"] = gpu_memory_utilization
            if max_model_len is not None:
                vllm_config["max_model_len"] = max_model_len

            return VLLMLlm(
                model_name=model_name,
                hf_token=hf_token,
                hf_cache=hf_cache,
                params=params,
                uses_api=uses_api,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
                max_model_len=vllm_config.get("max_model_len", max_model_len),
                trust_remote_code=trust_remote_code,
                vllm_config=vllm_config,  # Pass full config to VLLMLlm
            )

        else:
            raise ValueError(
                f"Unknown provider specified ({provider})."
                "Current support for azure, huggingface, and vllm only"
            )
