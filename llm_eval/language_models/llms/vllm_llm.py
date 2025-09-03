"""Support for vLLM inference engine for faster HuggingFace model serving."""

import gc
import logging
import os
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

from llm_eval.language_models.llms.base import BaseLLM
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.utils.exceptions import UnsupportedModelError


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

    def _load_model(self, pause_tracker: bool = True):
        """Load model using vLLM."""
        logging.info(f"Loading {self.model_name} with vLLM")
        if self.tracker and pause_tracker:
            self.tracker.stop()

        if self.model_name not in MODEL_MAPPING:
            raise UnsupportedModelError(self.model_name, MODEL_MAPPING.keys())

        model_id = self.model_config["id"]
        loading_kwargs = self.model_config["kwargs"].get("loading", {})
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
        self.tokenizer = get_tokenizer(model_id, tokenizer_mode="auto")

        if self.tracker and pause_tracker:
            self.tracker.start()

    def _create_sampling_params(self) -> SamplingParams:
        """
        Create vLLM SamplingParams from the model parameters.

        Maps common generation parameters to vLLM SamplingParams.
        """
        # Default vLLM sampling parameters
        vllm_params = {
            "temperature": 0.0,  # Greedy by default
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 512,
            "stop": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Map HuggingFace parameters to vLLM parameters
        param_mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_new_tokens": "max_tokens",
            "max_length": "max_tokens",
            "repetition_penalty": "frequency_penalty",
            "do_sample": None,  # Handled via temperature
        }

        for hf_param, vllm_param in param_mapping.items():
            if hf_param in self.params and vllm_param is not None:
                if hf_param == "repetition_penalty":
                    # Convert repetition penalty to frequency penalty
                    vllm_params[vllm_param] = self.params[hf_param] - 1.0
                else:
                    vllm_params[vllm_param] = self.params[hf_param]

        # Handle do_sample parameter
        if "do_sample" in self.params:
            if not self.params["do_sample"]:
                vllm_params["temperature"] = 0.0  # Force greedy sampling
        return SamplingParams(**vllm_params)

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt using chat template if available.

        Args:
            prompt: The user prompt

        Returns:
            Formatted prompt string
        """
        # Build conversation
        if self.system_prompt:
            conversation = [{"role": "system", "content": self.system_prompt}]
        else:
            conversation = []
        conversation.append({"role": "user", "content": prompt})

        # Try to use tokenizer's chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                template_kwargs = self.model_config["kwargs"].get("template", {})
                formatted_prompt = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                return formatted_prompt
            except Exception as e:
                raise Exception(f"Failed to apply chat template: {e}")

    def _prompt(
        self,
        prompt: str,
        context: Optional[str] = None,
        system: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> str:
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

        # Format the prompt
        formatted_prompt = self._format_prompt(prompt)

        # Create sampling parameters
        sampling_params = self._create_sampling_params()

        # Generate response
        outputs = self.model.generate([formatted_prompt], sampling_params)

        if not outputs or not outputs[0].outputs:
            return ""

        # Extract the generated text
        response = outputs[0].outputs[0].text

        return response

    def generate_batch(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in a batch.

        Args:
            prompts: List of prompts to generate responses for
            sampling_params: vLLM sampling parameters

        Returns:
            List of generated responses
        """
        if not self.model:
            self._load_model(pause_tracker=True)

        if sampling_params is None:
            sampling_params = self._create_sampling_params()

        # Format all prompts
        formatted_prompts = [self._format_prompt(prompt) for prompt in prompts]

        # Generate responses in batch
        outputs = self.model.generate(formatted_prompts, sampling_params)

        # Extract responses
        responses = []
        for output in outputs:
            if output.outputs:
                responses.append(output.outputs[0].text)
            else:
                responses.append("")

        return responses

    def unload_model(self):
        """Unload model to free up memory."""
        logging.info(f"Unloading {self.model_name}")
        if hasattr(self, "model") and self.model is not None:
            # vLLM doesn't have an explicit unload method, so we delete the object
            del self.model
        self.model = None
        self.tokenizer = None
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
