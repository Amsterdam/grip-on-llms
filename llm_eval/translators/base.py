"""
Module for handling of translators.
Currently supports LLM-based translations
as well as some HuggingFace models.
"""
from abc import abstractmethod


class BaseTranslator:
    """Base Translator class"""

    def __init__(self, model_name, source_lang="EN", target_lang="NL", params=dict):
        self._model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._params = params

    def translate(self, text, source_lang=None, target_lang=None):
        """Translate the given text."""
        # Possibly add carbon tracking here
        translation = self._translate(text, target_lang)
        return translation

    @abstractmethod
    def _translate(self, text, source_lang=None, target_lang=None):
        """Function to translate text should always be implemented"""
        raise NotImplementedError("Implement _translate function")

    @property
    def model_name(self):
        """Property to get the model name"""
        return self._model_name

    @property
    def params(self):
        """Property to get the model parameters"""
        return self._params

    def __call__(self, text, source_lang=None, target_lang=None):
        """Translate the given input.
        Args:
            text: The prompt to generate from.

        Returns:
            The translation as a string.
        """
        return self.translate(text, target_lang)

    def get_metadata(self):
        """Get model metadata for versioning purposes"""
        metadata = {
            "model_name": self.model_name,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "params": self.params,
        }
        return metadata
