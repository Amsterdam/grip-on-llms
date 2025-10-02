"""
Module for handling of LLMs and prompting them.
Currently supports the OpenAI models on Azure
as well as some HuggingFace models.
"""
import logging
from abc import abstractmethod
from typing import List

from codecarbon import OfflineEmissionsTracker

from llm_eval.utils.schemas import LLMMetadata, LLMResponse
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
        self._params = self.get_mapped_params(params)
        self.tracker = None
        self.uses_api = uses_api

    def prompt(self, prompt, context=None, system=None, response_format=None) -> LLMResponse:
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
            response.processed_response = clean_and_extract_multiple_choice(response.raw_response)
        else:
            response.processed_response = clean_and_extract_open_text_answers(
                response.raw_response
            )

        return response

    def process_batch(
        self, prompts, batch_size=None, context=None, system=None, response_format=None
    ) -> List[LLMResponse]:
        """Process a batch of prompts"""
        if self.tracker:
            self.tracker.start()

        # Generate the reposnse
        responses = self._process_batch(
            prompts=prompts,
            batch_size=batch_size,
            context=context,
            system=system,
            response_format=response_format,
        )

        # Stop tracker for the llm itself
        if self.tracker:
            self.tracker.stop()

        # If a specific format is desired, post-process accordingly
        for response in responses:
            if response_format == "multiple_choice":
                response.processed_response = clean_and_extract_multiple_choice(
                    response.raw_response
                )
            else:
                response.processed_response = clean_and_extract_open_text_answers(
                    response.raw_response
                )

        return responses

    def initialize_carbon_tracking(self, codecarbon_params=dict):
        """Tracks emissions offline using code carbon."""
        try:
            if not self.uses_api:
                self.tracker = OfflineEmissionsTracker(**codecarbon_params)
        except ValueError as e:
            logging.error(f"ValueError: {e}")
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
            logging.error(f"TrackerNotStartedError: {e}")
            return None

    def get_mapped_params(self, params):
        """Return mapped params if subclass defines map_params, else raw params"""
        if hasattr(self, "_map_params"):
            params = self._map_params(params)
        return params

    @abstractmethod
    def _prompt(self, prompt, context=None, system=None, response_format=None) -> LLMResponse:
        """Function to prompt model should always be implemented"""
        raise NotImplementedError("Implement _prompt function")

    @abstractmethod
    def _process_batch(
        self, prompts, batch_size=None, context=None, system=None, response_format=None
    ) -> List[LLMResponse]:
        """Function to batch prompt model should always be implemented"""
        raise NotImplementedError("Implement _process_batch function")

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

    @abstractmethod
    def _get_inference_engine(self):
        """Return the inference engine name: 'openai', 'huggingface', or 'vllm'"""
        raise NotImplementedError

    def _get_own_metadata(self):
        """To be overwritten in subclasses to add engine-specific metadata"""
        return {}

    def get_metadata(self):
        """Get model metadata for versioning purposes as LLMMetadata object"""
        return LLMMetadata(
            model_name=self.model_name,
            inference_engine=self._get_inference_engine(),
            params=serialize_params(self.params) if hasattr(self, "params") else None,
            **self._get_own_metadata(),
        )


def serialize_params(params):
    """Convert params to dictionary format."""
    if params is None:
        return None

    # Already a dict
    if isinstance(params, dict):
        return params

    # Pydantic model
    if hasattr(params, "model_dump"):
        return params.model_dump()
    if hasattr(params, "dict"):  # Pydantic v1
        return params.dict()

    # Last resort - try to convert to dict
    try:
        return dict(params)
    except (TypeError, ValueError):
        # If all else fails, return string representation
        return {"raw": str(params)}
