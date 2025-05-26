"""Support for locally-hosted HuggingFace models."""
import gc
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.language_models.llms.llm_templates import format_prompt
from llm_eval.utils.exceptions import UnsupportedModelError


class HuggingFaceLLM(BaseLLM):
    """A class to handle self-hosted HG models"""

    def __init__(self, model_name, hf_token, uses_api, hf_cache=None, params=dict):
        super().__init__(model_name, uses_api, params if params is not None else {})

        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

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

        model_config = MODEL_MAPPING[self.model_name]
        model_id = model_config["id"]
        kwargs = {
            "torch_dtype": torch.bfloat16,
            # "device_map": "auto",
            "token": self.hf_token,
        }
        kwargs.update(model_config["kwargs"])

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

        # conversation = [{"role": "user", "content": prompt}]
        # formatted_prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)

        formatted_prompt = format_prompt(prompt, model_name=self.model_name)

        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones(input_ids.shape).to(self.device)

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            # return_full_text=False,
            **self.params,
        )
        # fmt: off
        response = self.tokenizer.decode(
            output[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        # fmt: on
        response = response.replace(formatted_prompt.removeprefix("<s>"), "")
        response = response.removeprefix("assistant\n")

        return response

    def unload_model(self):
        """Unload model on demand to free up memory"""
        logging.info(f"Unloading {self.model_name}")
        self.model = None
        self.tokenizer = None
        gc.collect()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
