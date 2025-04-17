"""Support for locally-hosted HuggingFace models."""
import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_eval.translators.base import BaseTranslator
from llm_eval.translators.translator_config import TRANSLATOR_MAPPING
from llm_eval.utils.exceptions import UnsupportedModelError


class HuggingFaceTranslator(BaseTranslator):
    """A class to handle self-hosted HG models"""

    def __init__(self, model_name, source_lang, target_lang, hf_token, hf_cache=None, params=dict):
        super().__init__(
            model_name, source_lang, target_lang, params if params is not None else {}
        )

        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.model = None
        self.tokenizer = None

        self.source_id = self.get_lang_code(self.source_lang)
        self.target_id = self.get_lang_code(self.target_lang)

    def _load_model(self):
        """
        Load HF model based on a short model name.
        Expects known mapping to full model ID & params
        """
        logging.info(f"Loading {self.model_name}")
        if self.model_name not in TRANSLATOR_MAPPING:
            raise UnsupportedModelError(self.model_name, TRANSLATOR_MAPPING.keys())

        model_config = TRANSLATOR_MAPPING[self.model_name]
        model_id = model_config["id"]
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "token": self.hf_token,
        }
        kwargs.update(model_config["kwargs"])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, cache_dir=self.hf_cache, **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            src_lang=self.source_id,
            tgt_lang=self.target_id,
            cache_dir=self.hf_cache,
            **kwargs,
        )

    def get_lang_code(self, lang):
        """Get the correct language code"""
        return TRANSLATOR_MAPPING[self.model_name]["language_codes"][lang]

    def _translate(self, text, source_lang=None, target_lang=None, max_length=512, **kwargs):
        """Translate text"""
        if not self.model:
            self._load_model()

        if source_lang:
            logging.warning(
                f"Unable to use source_lang={source_lang}. Using default {self.source_lang}"
            )

        target_id = self.get_lang_code(target_lang) if target_lang else self.target_id

        device = get_device()
        inputs = self.tokenizer(text, return_tensors="pt").to(device)

        if self.model_name.startswith("nllb"):
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_id)
        elif self.model_name.startswith("mbart"):
            forced_bos_token_id = self.tokenizer.lang_code_to_id[target_id]
        else:
            raise UnsupportedModelError(self.model_name, TRANSLATOR_MAPPING.keys())

        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_length=max_length
        )
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
