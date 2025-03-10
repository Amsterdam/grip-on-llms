"""
Module for handling of LLMs and prompting them.
Currently supports the OpenAI models on Azure
as well as some HuggingFace models.
"""
from abc import abstractmethod


class BaseLLM:
    """Base LLM class"""

    def __init__(self, model_name, params):
        self._model_name = model_name
        self._params = params

    @abstractmethod
    def prompt(self, prompt, context=None, system=None, response_format=None):
        """Function to prompt model should always be implemented"""
        raise NotImplementedError("Implement prompt function")

    @property
    def model_name(self):
        """Property to get the model name"""
        return self._model_name

    @property
    def params(self):
        """Property to get the model parameters"""
        return self._params

    def __call__(self, prompt):
        """Run the LLM on the given input.
        Args:
            prompt: The prompt to generate from.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        return self.prompt(prompt)

    def get_metadata(self):
        """Get model metadata for versioning purposes"""
        metadata = {
            "model_name": self.model_name,
            "params": self.params,
        }
        return metadata
