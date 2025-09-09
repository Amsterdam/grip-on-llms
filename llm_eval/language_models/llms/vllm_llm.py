"""Support for vLLM inference engine for faster HuggingFace model serving."""

import gc
import logging
import os
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.language_models.llms.chat_template import create_chat_handler
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.utils.exceptions import UnsupportedModelError
from llm_eval.utils.string_utils import LLMResponse


class VLLMLlm(BaseLLM):
    """A class to handle HuggingFace models via vLLM for accelerated inference."""

    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        uses_api: bool = False,
        hf_cache: Optional[str] = None,
        params: Optional[Dict] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        vllm_config: Optional[Dict] = None,  # H100 optimized config
    ):
        """
        Initialize vLLM model.

        Args:
            model_name: Short name of the model (must be in MODEL_MAPPING)
            hf_token: HuggingFace token for private models
            uses_api: Whether this uses an API (always False for vLLM)
            hf_cache: Cache directory for HuggingFace models
            params: Generation parameters
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum model sequence length
            trust_remote_code: Whether to trust remote code in model
            vllm_config: H100-optimized vLLM configuration
        """
        super().__init__(model_name, uses_api, params if params is not None else {})

        self.hf_token = hf_token
        self.hf_cache = hf_cache
        self.model = None
        self.tokenizer = None
        self.system_prompt = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.vllm_config = vllm_config or {}  # Store H100 configuration
        self.model_config = MODEL_MAPPING[self.model_name]
        self.chat_handler = None

    def _load_model(self, pause_tracker: bool = True):  # noqa
        """Load model using vLLM."""
        logging.info(f"Loading {self.model_name} with vLLM")
        if self.tracker and pause_tracker:
            self.tracker.stop()

        if self.model_name not in MODEL_MAPPING:
            raise UnsupportedModelError(self.model_name, MODEL_MAPPING.keys())

        model_id = self.model_config["id"]

        loading_kwargs = self.model_config["kwargs"].get("loading", {})
        template_kwargs = self.model_config["kwargs"].get("template", {})

        self.tokenizer = get_tokenizer(model_id)

        # With tokenizer initialized, set all eos tokens and params
        self._set_eos_tokens()

        if "mistral" in model_id.lower():
            loading_kwargs["tokenizer_mode"] = "mistral"

        self.system_prompt = loading_kwargs.pop("system_prompt", None)

        # Start with H100-optimized configuration
        vllm_kwargs = self.vllm_config.copy()

        # Override with basic initialization arguments
        vllm_kwargs.update(
            {
                "model": model_id,
                "tokenizer": model_id,
                "tensor_parallel_size": self.tensor_parallel_size,
                "trust_remote_code": self.trust_remote_code,
                "download_dir": self.hf_cache,
            }
        )

        # Override with user-specified parameters
        if hasattr(self, "gpu_memory_utilization"):
            vllm_kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        if self.max_model_len is not None:
            vllm_kwargs["max_model_len"] = self.max_model_len

        # Add HuggingFace token if provided
        if self.hf_token:
            # vLLM uses HUGGING_FACE_HUB_TOKEN environment variable
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

        # Override with model-specific loading parameters
        model_loading_kwargs = {
            k: v for k, v in loading_kwargs.items() if k not in ["system_prompt"]
        }

        vllm_kwargs.update(model_loading_kwargs)

        # Remove any None values and ensure proper types
        vllm_kwargs = {k: v for k, v in vllm_kwargs.items() if v is not None}

        self.model = LLM(**vllm_kwargs)

        self.chat_handler = create_chat_handler(
            "vllm", self.tokenizer, None, self.system_prompt, template_kwargs
        )

        if self.tracker and pause_tracker:
            self.tracker.start()

    def _map_params(self, params) -> SamplingParams:
        """
        Create vLLM SamplingParams from the model parameters.
        Maps common generation parameters to vLLM SamplingParams.
        """
        # These are expected to be explicitly passed
        vllm_params = {
            "skip_special_tokens": True,
            "include_stop_str_in_output": False,
            # "truncate_prompt_tokens": True,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
            "max_tokens": params.get("max_new_tokens") or params.get("max_length"),
        }

        # Map repetition_penalty → frequency_penalty approximation
        if "repetition_penalty" in params:
            vllm_params["frequency_penalty"] = params["repetition_penalty"] - 1.0
        # frequency/presence_penalty map directly if provided
        if "frequency_penalty" in params:
            vllm_params["frequency_penalty"] = params["frequency_penalty"]
        if "presence_penalty" in params:
            vllm_params["presence_penalty"] = params["presence_penalty"]

        # Handle do_sample parameter
        if "do_sample" in params and not params["do_sample"]:
            vllm_params["temperature"] = 0.0

        return SamplingParams(**vllm_params)

    def _set_eos_tokens(self):
        """Set EOS tokens and IDs in sampling params for generation"""
        # With tokenizer initialized, adjust EOS
        eos_str = self.tokenizer.decode([self.tokenizer.eos_token_id])

        basic_special_tokens = list(getattr(self.tokenizer, "all_special_tokens", []))
        additional_special_tokens = list(getattr(self.tokenizer, "additional_special_tokens", []))
        known_special_tokens = ["[INST]", "###", "\n\n[INST]"]

        # Combine all stop strings, filter to avoid None or empty
        all_tokens = (
            [eos_str] + additional_special_tokens + basic_special_tokens + known_special_tokens
        )

        # Convert all stop strings into token IDs
        stop_token_ids = []
        for s in all_tokens:
            token_ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(token_ids) == 1:  # Only keep single-token stops
                stop_token_ids.append(token_ids[0])

        self.params.stop = (self.params.stop or []) + all_tokens
        self.params.stop_token_ids = stop_token_ids

    def _prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        system: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        """
        Prompt model using vLLM.

        Args:
            prompt: The user prompt
            context: Additional context (unused in current implementation)
            system: System prompt override (unused in current implementation)
            response_format: Expected response format (unused in current implementation)

        Returns:
            Generated response as string
        """
        if not self.model:
            self._load_model(pause_tracker=True)

        response = LLMResponse()
        response.raw_prompt = prompt
        # Format the prompt
        conversation = self.chat_handler.format_conversation(prompt, context, system)
        formatted_prompt = self.chat_handler.apply_chat_template_for_generation(conversation)
        response.formatted_prompt = formatted_prompt

        # Generate response
        outputs = self.model.generate(formatted_prompt, self.params)

        if not outputs or not outputs[0].outputs:
            response.error = True
            response.exception = "Empty response"
            return response

        # Extract the generated text
        response.raw_response = outputs[0].outputs[0].text
        return response

    def _process_batch(
        self,
        prompts: List[str],
        batch_size: Optional[int] = None,
        context: Optional[str] = None,
        system: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts in a batch.

        Args:
            prompts: List of prompts to generate responses for
            batch_size: In what size the data should be processed
            context: Additional context (unused in current implementation)
            system: System prompt override (unused in current implementation)
            response_format: Expected response format (unused in current implementation)
        Returns:
            List of generated responses
        """
        if not self.model:
            self._load_model(pause_tracker=True)

        # Format all prompts
        conversations = [
            self.chat_handler.format_conversation(prompt, context, system) for prompt in prompts
        ]
        formatted_prompts = [
            self.chat_handler.apply_chat_template_for_generation(conv) for conv in conversations
        ]

        # Generate responses in batch
        if batch_size is None:
            outputs = self.model.generate(formatted_prompts, self.params)
        else:
            outputs = []
            batch = []
            for prompt in formatted_prompts:
                batch.append(prompt)
                if len(batch) == batch_size:
                    outputs.extend(self.model.generate(batch, self.params))
                    batch = []
            # flush
            if batch:
                outputs.extend(self.model.generate(batch, self.params))

        # Extract responses
        responses = []
        for i, output in enumerate(outputs):
            response = LLMResponse()
            response.raw_prompt = prompts[i]
            response.formatted_prompt = formatted_prompts[i]
            if output.outputs:
                response.raw_response = output.outputs[0].text
            else:
                response.error = True
                response.exception = "Empty response"
                response.raw_response = ""
            responses.append(response)
        return responses

    def unload_model(self):
        """Unload model to free up memory."""
        logging.info(f"Unloading {self.model_name}")
        if hasattr(self, "model") and self.model is not None:
            # vLLM doesn't have an explicit unload method, so we delete the object
            del self.model
        self.model = None
        self.tokenizer = None
        self.chat_handler = None
        gc.collect()

    def get_metadata(self):
        """Get model metadata including vLLM-specific information."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "inference_engine": "vllm",
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
            }
        )
        return metadata
