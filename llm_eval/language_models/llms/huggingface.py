"""Support for locally-hosted HuggingFace models."""
import logging
from typing import List

import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.language_models.llms.chat_template import create_chat_handler
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.language_models.llms.llm_utils import aggressive_gpu_cleanup
from llm_eval.utils.exceptions import UnsupportedModelError
from llm_eval.utils.string_utils import LLMResponse

torch._dynamo.disable()


class HuggingFaceLLM(BaseLLM):
    """A class to handle self-hosted HG models"""

    def __init__(self, model_name, hf_token, uses_api, hf_cache=None, params=dict):
        super().__init__(model_name, uses_api, params if params is not None else {})

        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.system_prompt = None
        self.model_config = MODEL_MAPPING[self.model_name]
        self.chat_handler = None

    def _load_model(self, pause_tracker=True):
        """
        Load HF model based on a short model name.
        Expects known mapping to full model ID & params
        """
        logging.info(f"Loading {self.model_name}")
        if self.tracker and pause_tracker:
            self.tracker.stop()

        if self.model_name not in MODEL_MAPPING:
            raise UnsupportedModelError(self.model_name, MODEL_MAPPING.keys())

        model_id = self.model_config["id"]
        kwargs = {
            "torch_dtype": torch.bfloat16,
            # "device_map": "auto",
            "token": self.hf_token,
        }
        kwargs.update(self.model_config["kwargs"].get("loading", {}))
        self.system_prompt = kwargs.pop("system_prompt", None)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=self.hf_cache, **kwargs
        )
        self.device = get_device()
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.hf_cache, **kwargs)

        template_kwargs = self.model_config["kwargs"].get("template", {})
        self.chat_handler = create_chat_handler(
            engine="huggingface",
            tokenizer=self.tokenizer,
            device=self.device,
            system_prompt=self.system_prompt,
            template_kwargs=template_kwargs,
        )

        if self.tracker and pause_tracker:
            self.tracker.start()

    def _map_params(self, params):
        return {
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "max_new_tokens": params.get("max_new_tokens"),
            "repetition_penalty": params.get("repetition_penalty"),
            "do_sample": params.get("do_sample"),
        }

    def _prompt(self, prompt, context=None, system=None, force_format=None):
        """Prompt model by optionally providing a custom system prompt or context"""
        if not self.model:
            self._load_model(pause_tracker=True)

        response = LLMResponse()
        response.raw_prompt = prompt

        conversation = self.chat_handler.format_conversation(prompt, context, system)
        response.formatted_prompt = self.chat_handler.apply_chat_template_for_display(conversation)

        generation_input = self.chat_handler.apply_chat_template_for_generation(conversation)

        output = self.model.generate(
            generation_input["input_ids"],
            attention_mask=generation_input["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            # return_full_text=False,
            **self.params,
        )
        output_text = self.tokenizer.decode(
            output[0][generation_input["input_length"] :], skip_special_tokens=True
        )
        response.raw_response = output_text
        return response

    def _process_batch(
        self, prompts, batch_size=None, context=None, system=None, response_format=None
    ) -> List[LLMResponse]:
        """Process a batch of prompts"""
        return [self._prompt(prompt, context, system, response_format) for prompt in prompts]

    def unload_model(self):
        """Unload model on demand to free up memory"""
        logging.info(f"Unloading {self.model_name}")

        # Clear model and tokenizer references
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, "chat_handler") and self.chat_handler is not None:
            del self.chat_handler

        self.model = None
        self.tokenizer = None
        self.chat_handler = None

        aggressive_gpu_cleanup()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
