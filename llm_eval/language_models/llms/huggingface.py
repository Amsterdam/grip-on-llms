"""Support for locally-hosted HuggingFace models."""
import logging
import re

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

    def _load_model(self):
        """
        Load HF model based on a short model name.
        Expects known mapping to full model ID & params
        """
        logging.info(f"Loading {self.model_name}")
        if self.model_name not in MODEL_MAPPING:
            raise UnsupportedModelError(self.model_name, MODEL_MAPPING.keys())

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

    def clean_and_extract(self, input_string):
        """Clean response string using regex."""
        # Remove everything between [] and <> (including the brackets)
        cleaned_string = re.sub(r"\[.*?\]|\<.*?\>", "", input_string)

        # Check if the string contains "Answer:"
        answer_match = re.search(r"\bAnswer:\s*([A-D])\.", cleaned_string.strip())

        if answer_match:
            # If "Answer:" is found, return the letter
            return answer_match.group(1)

        # If no "Answer:" is found, extract the first valid letter
        match = re.match(r"^\s*([A-D])\.\s*", cleaned_string.strip())

        if match:
            return match.group(1)

        return input_string

    def _prompt(self, prompt, context=None, system=None, force_format=None):
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

        response = self.clean_and_extract(response)
        return response


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
