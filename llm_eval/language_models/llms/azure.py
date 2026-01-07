"""Support for Azure deployments of models"""
import logging
import time
from typing import List

from openai import AzureOpenAI
from tqdm import tqdm

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.utils.schemas import LLMResponse
from llm_eval.utils.setup_utils import get_gpt_secrets


class AzureLLM(BaseLLM):
    """
    A class to support use of LLMs using Azure endpoints.
    Expects Azure deployment and corresponding endpoint, key, etc.
    """

    def __init__(self, model_name, api_endpoint, api_key, api_version, uses_api, params=dict):
        super().__init__(model_name, uses_api, params if params is not None else {})

        self._reset_credentials(api_endpoint, api_key, api_version)

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

    def _reset_credentials(self, api_endpoint=None, api_key=None, api_version=None):
        logging.info(f"(Re)setting {self.model_name} credentials.")

        if not all([api_endpoint, api_key, api_version]):
            credentials = get_gpt_secrets()
            api_endpoint = credentials["API_ENDPOINT"]
            api_key = credentials["API_KEY"]
            api_version = credentials["API_VERSION"]

        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.client = self._get_client()

        self.prompt_count_credentials = 0

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

    def _make_api_call(self, prompt, context, system, force_format):
        response = LLMResponse()
        response.raw_prompt = prompt

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

        if response.raw_response == "":
            logging.warning("Empty response!")
            # from pprint import pprint
            # pprint(api_response)

        return response

    def _prompt(
        self,
        prompt,
        context=None,
        system=None,
        force_format=None,
        limit_requests=False,
        renew_credentials_after=100,
        max_retries=3,
    ):
        """Prompt model by optionally providing a custom system prompt or context"""
        if not self.client:
            self.client = self._get_client()

        if renew_credentials_after and (self.prompt_count_credentials > renew_credentials_after):
            self._reset_credentials()

        if limit_requests:
            self._limit_requests(prompt)

        for attempt in range(max_retries):
            try:
                response = self._make_api_call(prompt, context, system, force_format)
                break

            except Exception as e:
                error_str = str(e)
                logging.error(f"{self.model_name} failed: {error_str}")

                # Allowed reasons to retry: currently only 401s
                if "401" in error_str and attempt < max_retries - 1:
                    logging.warning(f"Renewing credentials (attempt {attempt + 1}/{max_retries})")
                    self._reset_credentials()
                    continue

                # in all other cases or if max_retries are reached -> no retrying, just move on
                response = LLMResponse()
                response.raw_prompt = prompt
                response.exception = error_str
                response.error = True
                break

        self.prompt_count_credentials += 1

        return response

    def _process_batch(
        self, prompts, batch_size=None, context=None, system=None, response_format=None
    ) -> List[LLMResponse]:
        """Process a batch of prompts"""
        return [self._prompt(prompt, context, system, response_format) for prompt in tqdm(prompts)]

    def unload_model(self):
        """Unload model on demand to free up memory and reduce resource usage"""
        pass

    def _get_inference_engine(self):
        return "azure"

    def _get_own_metadata(self):
        """Get Azure-specific information."""
        return {
            "api_version": self.api_version,
        }
