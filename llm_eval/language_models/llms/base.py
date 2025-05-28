"""
Module for handling of LLMs and prompting them.
Currently supports the OpenAI models on Azure
as well as some HuggingFace models.
"""
from abc import abstractmethod

from codecarbon import OfflineEmissionsTracker

from llm_eval.utils.string_utils import (
    clean_and_extract_multiple_choice,
    clean_and_extract_open_text_answers,
)


class TrackerNotStartedError(Exception):
    """Raises error whenever tracker is not started."""

    def __init__(self, message):
        super().__init__(message)


class BaseLLM:
    """Base LLM class"""

    def __init__(self, model_name, uses_api, params):
        self._model_name = model_name
        self._params = params
        self.tracker = None
        self.uses_api = uses_api

    def prompt(self, prompt, context=None, system=None, response_format=None):
        """Starts and stops code carbon tracker, and gets response from model."""
        # Start carbon tracker
        if self.tracker:
            self.tracker.start()

        # Generate the reposnse
        response = self._prompt(prompt, context, system, response_format)

        # Stop tracker for the llm itself
        if self.tracker:
            self.tracker.stop()

        # If a specific format is desired, post-process accordingly
        if response_format == "multiple_choice":
            response = clean_and_extract_multiple_choice(response)
        else:
            response = clean_and_extract_open_text_answers(response)

        return response

    def initialize_carbon_tracking(self, codecarbon_params=dict):
        """Tracks emissions offline using code carbon."""
        try:
            if not self.uses_api:
                self.tracker = OfflineEmissionsTracker(**codecarbon_params)
        except ValueError as e:
            print(f"ValueError: {e}")
            return None

    def get_carbon_data(self):
        """Get code carbon tracker data and return."""
        try:
            if not self.tracker:
                raise TrackerNotStartedError(
                    "Exception raised when the tracker has not been started."
                )
            final_results = self.tracker.final_emissions_data.__dict__
            return final_results
        except TrackerNotStartedError as e:
            print(f"TrackerNotStartedError: {e}")
            return None

    @abstractmethod
    def _prompt(self, prompt, context=None, system=None, response_format=None):
        """Function to prompt model should always be implemented"""
        raise NotImplementedError("Implement _prompt function")

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

    @abstractmethod
    def unload_model(self):
        """Unload model on demand to free up memory and reduce resource usage"""
        raise NotImplementedError("Implement unload_model function")

    def get_metadata(self):
        """Get model metadata for versioning purposes"""
        metadata = {
            "model_name": self.model_name,
            "params": self.params,
        }
        return metadata
