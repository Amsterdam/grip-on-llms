"""
Functionality for routing to different LLMs.
Currently supports Azure OpenAI deployments and HuggingFace models.

Usage Example:
    model = LLMRouter.get_model(provider="huggingface", model_name="falcon-7b")
"""
import logging

from llm_eval.language_models.llms.huggingface import HuggingFaceLLM
from llm_eval.language_models.llms.openai import OpenAILLM


class LLMRouter:
    """Route LLMs depending on model and desired provider."""

    @staticmethod
    def get_model(
        provider,
        model_name,
        api_endpoint=None,
        api_key=None,
        api_version=None,
        hf_token=None,
        hf_cache=None,
        params=None,
        uses_api=False,
    ):
        """Get corresponding LLM instance based on the specified provider and model.

        Args:
            provider (str): The provider of the model. Supported: "azure" & "huggingface".
            model_name (str): The name of the model to load.
            api_endpoint (str, optional): The endpoint for API-based models (e.g. Azure OpenAI).
            api_key (str, optional): The API key for authentication (for API-based models).
            api_version (str, optional): The API version for API-based models.
            hf_token (str, optional): The Hugging Face token for accessing private models.
            hf_cache (str, optional): Path to the local cache for Hugging Face models.
            params (dict, optional): Additional parameters for the model.

        Returns:
            An instance of `OpenAILLM` or `HuggingFaceLLM`.

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

        else:
            raise ValueError(
                f"Unknown provider specified ({provider})."
                "Current support for azure and huggingface only"
            )
