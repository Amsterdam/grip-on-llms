"""Schemas for LLM responses, benchmark results, validators, etc"""
from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class LLMResponse:
    """Storing the in and outputs of LLMs"""

    raw_prompt: str = ""
    formatted_prompt: Union[str, List] = ""
    raw_response: str = ""
    error: bool = False
    processed_response: Optional[str] = None
    exception: Optional[str] = None
