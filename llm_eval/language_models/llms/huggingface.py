"""Support for locally-hosted HuggingFace models."""
import gc
import logging
from typing import List

import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
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

        if self.tracker and pause_tracker:
            self.tracker.start()

    def _prompt(self, prompt, context=None, system=None, force_format=None):
        """Prompt model by optionally providing a custom system prompt or context"""
        if not self.model:
            self._load_model(pause_tracker=True)

        response = LLMResponse()
        response.raw_prompt = prompt
        if self.system_prompt:
            conversation = [{"role": "system", "content": self.system_prompt}]
        else:
            conversation = []
        conversation.append([{"role": "user", "content": prompt}])
        response.formatted_prompt = conversation

        template_kwargs = self.model_config["kwargs"].get("template", {})

        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.device)

        attention_mask = torch.ones(input_ids.shape).to(self.device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            # return_full_text=False,
            **self.params,
        )
        output_text = self.tokenizer.decode(
            output[0][input_ids.shape[-1] :], skip_special_tokens=True
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
        self.model = None
        self.tokenizer = None
        gc.collect()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
