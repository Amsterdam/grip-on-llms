"""
Functionality for routing to different Translators.
Currently supports LLM-based translators and HuggingFace models.

Usage Example:
    translator = TranslatorRouter.get_translator(
        translator_type="huggingface",
        model_name="nllb-1b",
        source_lang="EN",
        target_lang="NL",
    )
"""
import logging

from llm_eval.translators.huggingface import HuggingFaceTranslator
from llm_eval.translators.llm_based import LLMTranslator


class TranslatorRouter:
    """Route translators depending on model and desired type."""

    @staticmethod
    def get_translator(
        translator_type,
        model_name,
        source_lang,
        target_lang,
        llm=None,
        hf_token=None,
        hf_cache=None,
        params=None,
    ):
        """Get corresponding Translator instance based on the specified type and model.

        Args:
            translator_type (str): Type of the translator. Supported: "llm_based" & "huggingface".
            model_name (str): The name of the model to load.
            source_lang (str): The source language ("EN", "NL")
            target_lang (str): The target language ("EN", "NL")
            llm (LLM, optional): An LLM instance for llm_based translators.
            hf_token (str, optional): The Hugging Face token for accessing private models.
            hf_cache (str, optional): Path to the local cache for Hugging Face models.
            params (dict, optional): Additional parameters for the model.

        Returns:
            An instance of `LLMTranslator` or `HuggingFaceTranslator`.

        Raises:
            ValueError: If an unknown translator_type is specified.
        """
        logging.info(f"Getting a model. Provider: {translator_type}; Model: {model_name}")

        if translator_type == "llm_based":
            return LLMTranslator(
                model_name=model_name,
                source_lang=source_lang,
                target_lang=target_lang,
                llm=llm,
                params=params,
            )

        elif translator_type == "huggingface":
            return HuggingFaceTranslator(
                model_name=model_name,
                source_lang=source_lang,
                target_lang=target_lang,
                hf_token=hf_token,
                hf_cache=hf_cache,
                params=params,
            )

        else:
            raise ValueError(
                f"Unknown translator type specified ({translator_type})."
                "Current support for llm_based and huggingface only"
            )
