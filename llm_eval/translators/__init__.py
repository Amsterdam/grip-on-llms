"""
Currently, we support HuggingFace models as well as llm-based translations.
Translators take a text as input and optionally desired source and target languages
(if different for the current text than the default languages of the translator)
and return the translation in the form of a string.

In the future, we could also support e.g. translators library or other providers.

A dedicated TranslatorRouter can be used to instantiate the corresponding translator.
"""
from .huggingface import HuggingFaceTranslator
from .llm_based import LLMTranslator
from .translator_router import TranslatorRouter

all = ["HuggingFaceTranslatorceLLM", "LLMTranslator", "TranslatorRouter"]
