"""Support for OpenAI models"""
import logging

from openai import AzureOpenAI

from llm_eval.language_models.llms.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    A class to support use of OpenAI LLMs.
    Expects Azure deployment and corresponding endpoint, key, etc.
    """

    def __init__(self, model_name, api_endpoint, api_key, api_version, params=dict):
        super().__init__(model_name, params if params is not None else {})

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

    def prompt(self, prompt, context=None, system=None, force_format=None):
        """Prompt model by optionally providing a custom system prompt or context"""
        if not self.client:
            self.client = self._get_client()

        conversation = []

        if system:
            conversation.append({"role": "system", "content": system})

        if context:
            conversation.append({"role": "system", "content": context})

        conversation.append({"role": "user", "content": prompt})

        if force_format:
            if force_format == "json":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conversation,
                    **self.params,
                    response_format={"type": "json_object"},
                )

            else:
                raise NotImplementedError(
                    "Currently there is no support for special formats other than json"
                )

        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation,
                **self.params,
            )

        finish_reason = response.choices[0].finish_reason
        if finish_reason != "stop":
            logging.info(f"Finish reason: {finish_reason}")

        return response.choices[0].message.content
