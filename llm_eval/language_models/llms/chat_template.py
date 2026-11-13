"""Unified chat template handling for HuggingFace and vLLM implementations."""

import json
import logging
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


class ChatTemplateHandler:
    """
    Unified chat template handling for consistent prompt formatting
    across inference engines.
    """

    def __init__(
        self,
        tokenizer,
        system_prompt: Optional[str] = None,
        template_kwargs: Optional[Dict] = None,
        model_id: Optional[str] = None,
    ):
        """
        Initialize the chat template handler.

        Args:
            tokenizer: Either HuggingFace tokenizer or vLLM tokenizer
            system_prompt: Optional system prompt to prepend to conversations
            template_kwargs: Additional kwargs for chat template application
            model_id: HF repo id, used to find a chat template the tokenizer does not have
        """
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.template_kwargs = template_kwargs or {}
        self.force_no_thinking = self.template_kwargs.pop("force_no_thinking", False)

        if model_id and getattr(self.tokenizer, "chat_template", None) is None:
            self._load_processor_chat_template(model_id)

    def _load_processor_chat_template(self, model_id: str):
        """
        Some repos (e.g. Mistral Small 3.1) only ship their chat template in
        chat_template.json, which tokenizers do not read. Load it from there.
        """
        try:
            template_path = hf_hub_download(model_id, "chat_template.json")
        except EntryNotFoundError:
            return

        with open(template_path, encoding="utf-8") as f:
            self.tokenizer.chat_template = json.load(f)["chat_template"]
        logging.info(f"Loaded chat template for {model_id} from chat_template.json")

    def format_conversation(
        self, prompt: str, context: Optional[str] = None, system: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Create a standardized conversation format.

        Args:
            prompt: User prompt
            context: Additional context (can be incorporated into prompt)
            system: Override system prompt for this conversation

        Returns:
            Standardized conversation list
        """
        conversation = []

        # Use override system prompt, instance system prompt, or none
        effective_system = system or self.system_prompt
        if effective_system:
            conversation.append({"role": "system", "content": effective_system})

        # Incorporate context into prompt if provided
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt

        conversation.append({"role": "user", "content": full_prompt})

        return conversation

    def apply_chat_template_for_generation(
        self, conversation: List[Dict[str, str]]
    ) -> Union[str, Any]:
        """
        Apply chat template for text generation.
        Returns formatted string for vLLM or tokenized tensors for HuggingFace.

        This method should be overridden by engine-specific implementations.
        """
        raise NotImplementedError("Must be implemented by engine-specific handler")


class HuggingFaceChatHandler(ChatTemplateHandler):
    """HuggingFace-specific chat template handler."""

    def __init__(
        self,
        tokenizer,
        device,
        system_prompt: Optional[str] = None,
        template_kwargs: Optional[Dict] = None,
        model_id: Optional[str] = None,
    ):
        super().__init__(tokenizer, system_prompt, template_kwargs, model_id)
        self.device = device

    def apply_chat_template_for_generation(self, conversation: List[Dict[str, str]]):
        """Apply chat template and return tokenized input for HuggingFace."""
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            # transformers >= 5 returns a dict by default
            return_dict=False,
            **self.template_kwargs,
        ).to(self.device)

        attention_mask = torch.ones(input_ids.shape).to(self.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_length": input_ids.shape[-1],  # Store for response extraction
        }

    def apply_chat_template_for_display(self, conversation: List[Dict[str, str]]) -> str:
        """Apply chat template and return formatted string for display/logging."""
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )


class VLLMChatHandler(ChatTemplateHandler):
    """vLLM-specific chat template handler."""

    def apply_chat_template_for_generation(self, conversation: List[Dict[str, str]]) -> str:
        """Apply chat template and return formatted string for vLLM."""
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_kwargs,
        )
        if self.force_no_thinking:
            no_think_suffix = (
                "\n</think>" if "<think>" in formatted_prompt else "<think>\n</think>"
            )
            formatted_prompt += no_think_suffix

        return formatted_prompt

    def apply_chat_template_for_display(self, conversation: List[Dict[str, str]]) -> str:
        """Apply chat template and return formatted string for display/logging."""
        return self.apply_chat_template_for_generation(conversation)


# Factory function for creating appropriate handler
def create_chat_handler(
    engine: str,
    tokenizer,
    device=None,
    system_prompt: Optional[str] = None,
    template_kwargs: Optional[Dict] = None,
    model_id: Optional[str] = None,
) -> ChatTemplateHandler:
    """
    Factory function to create appropriate chat handler.

    Args:
        engine: "huggingface" or "vllm"
        tokenizer: Tokenizer instance
        device: Device for HuggingFace (ignored for vLLM)
        system_prompt: Optional system prompt
        template_kwargs: Template-specific kwargs
        model_id: HF repo id, used to find a chat template the tokenizer does not have

    Returns:
        Appropriate ChatTemplateHandler instance
    """
    if engine.lower() == "huggingface":
        if device is None:
            raise ValueError("Device must be provided for HuggingFace handler")
        return HuggingFaceChatHandler(tokenizer, device, system_prompt, template_kwargs, model_id)
    elif engine.lower() == "vllm":
        return VLLMChatHandler(tokenizer, system_prompt, template_kwargs, model_id)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
