"""
Module for handling of LLMs and prompting them.
Currently supports the OpenAI models on Azure
as well as some HuggingFace models.
"""
import logging
import os
from abc import abstractmethod

import torch
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval.language_models.llm_config import MODEL_MAPPING
from llm_eval.language_models.llm_templates import format_prompt


class UnsupportedModelError(Exception):
    """Exception raised for unsupported models."""

    pass


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


class HuggingFaceLLM(BaseLLM):
    """A class to handle self-hosted HG models"""

    def __init__(self, model_name, hf_token, hf_cache=None, params=dict):
        super().__init__(model_name, params if params is not None else {})

        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """
        Load HF model based on a short model name.
        Expects known mapping to full model ID & params
        """
        logging.info(f"Loading {self.model_name}")
        if self.model_name not in MODEL_MAPPING:
            error_msg = f"Unsupported Model {self.model_name}. Choose from {MODEL_MAPPING.keys()}"
            raise UnsupportedModelError(error_msg)

        model_config = MODEL_MAPPING[self.model_name]
        model_id = model_config["id"]
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": self.hf_token,
        }
        kwargs.update(model_config["kwargs"])

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=self.hf_cache, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.hf_cache, **kwargs)

    def prompt(self, prompt, context=None, system=None, force_format=None):
        """Prompt model by optionally providing a custom system prompt or context"""
        if not self.model:
            self._load_model()

        # conversation = [{"role": "user", "content": prompt}]
        # formatted_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        formatted_prompt = format_prompt(prompt, model_name=self.model_name)

        device = get_device()
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape).to(device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            # return_full_text=False,
            **self.params,
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(
            formatted_prompt.removeprefix("<s>"), ""
        )

        return response


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


class LLMRouter:
    """Route LLMs depending on model and desired provider."""

    @staticmethod
    def get_model(
        provider,
        model_name="falcon",
        api_endpoint=None,
        api_key=None,
        api_version=None,
        hf_token=None,
        hf_cache=None,
        params=None,
    ):
        """Get corresponding based on provider and model name"""
        logging.info(f"Getting a model. Provider: {provider}; Model: {model_name}")

        if provider == "azure":
            if "gpt" in model_name:
                return OpenAILLM(
                    model_name=model_name,
                    api_endpoint=api_endpoint,
                    api_key=api_key,
                    api_version=api_version,
                    params=params,
                )
            else:
                raise NotImplementedError(
                    "Currently there is no support for models other than GPT on Azure."
                )
        elif provider == "huggingface":
            return HuggingFaceLLM(
                model_name=model_name, hf_token=hf_token, hf_cache=hf_cache, params=params
            )

        else:
            raise ValueError(
                f"Unknown provider specified ({provider})."
                "Current support for azure and huggingface only"
            )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        # "stop": None,
    }
    hf_params = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.65,
        # "top_k": 25,
        "max_new_tokens": 200,
        "no_repeat_ngram_size": 3,
        "num_return_sequences": 1,
    }

    # test = "Test!"
    test = "Hoe maak ik een melding in Amsterdam?"

    # Test GPT
    model = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        api_version=os.environ["API_VERSION"],
        params=gpt_params,
    )
    # print(f"GPT Response to {test}!: {model.prompt(test)}")
    # print(f"Second attempt {model(test)}")

    # Test HF Model
    model = LLMRouter.get_model(
        provider="huggingface",
        model_name="tiny-llama",
        hf_token=os.environ["HF_TOKEN"],
        #        hf_cache=os.environ["HF_CACHE"],
        hf_cache=None,
        params=hf_params,
    )
    print(f"HF Response to {test}!: {model.prompt(test)}")
