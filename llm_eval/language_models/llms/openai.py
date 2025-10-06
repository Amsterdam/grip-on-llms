"""Support for OpenAI models"""
import logging
import time
from typing import List

from openai import AzureOpenAI

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.utils.schemas import LLMResponse


class OpenAILLM(BaseLLM):
    """
    A class to support use of OpenAI LLMs.
    Expects Azure deployment and corresponding endpoint, key, etc.
    """

    def __init__(self, model_name, api_endpoint, api_key, api_version, uses_api, params=dict):
        super().__init__(model_name, uses_api, params if params is not None else {})

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.client = self._get_client()

    def _get_client(self):
        client = AzureOpenAI(
            azure_endpoint=self.api_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        return client

    def _limit_requests(self, prompt):
        limit_per_minute = 400000
        limit_per_second = limit_per_minute / 60
        time.sleep(len(prompt) / limit_per_second)

    def _get_api_response(self, conversation, force_format):
        if force_format:
            if force_format == "json":
                api_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    **self.params,
                    response_format={"type": "json_object"},
                )

            # Handled by base class
            elif force_format == "multiple_choice":
                api_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    **self.params,
                )

            else:
                raise NotImplementedError(
                    "Currently there is no support for special formats other than json"
                )

        else:
            api_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                **self.params,
            )

        return api_response

    def _get_formatted_conversation(self, prompt, context=None, system=None):
        conversation = []
        if system:
            conversation.append({"role": "system", "content": system})

        if context:
            conversation.append({"role": "system", "content": context})

        conversation.append({"role": "user", "content": prompt})
        return conversation

    def _prompt(self, prompt, context=None, system=None, force_format=None, limit_requests=False):
        """Prompt model by optionally providing a custom system prompt or context"""
        if not self.client:
            self.client = self._get_client()

        if limit_requests:
            self._limit_requests(prompt)

        response = LLMResponse()
        response.raw_prompt = prompt
        try:
            conversation = self._get_formatted_conversation(prompt, context, system)
            response.formatted_prompt = conversation

            api_response = self._get_api_response(conversation, force_format)

            finish_reason = api_response.choices[0].finish_reason
            if finish_reason != "stop":
                logging.info(f"Finish reason: {finish_reason}")

            # Handle non-stop finish reasons which we want to treat as exceptions
            if finish_reason == "content_filter":
                response.error = True
                response.exception = f"Request terminated with finish_reason: {finish_reason}"

            response.raw_response = api_response.choices[0].message.content or ""
        except Exception as e:
            response.exception = str(e)
            response.error = True
        return response

    def _process_batch(
        self, prompts, batch_size=None, context=None, system=None, response_format=None
    ) -> List[LLMResponse]:
        """Process a batch of prompts"""
        return [self._prompt(prompt, context, system, response_format) for prompt in prompts]

    def unload_model(self):
        """Unload model on demand to free up memory and reduce resource usage"""
        pass

    def _get_inference_engine(self):
        return "openai"

    def _get_own_metadata(self):
        """Get OpenAI-specific information."""
        return {
            "api_version": self.api_version,
        }
