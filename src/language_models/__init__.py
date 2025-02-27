"""
Currently we support HuggingFace models as well as Azure deployments of OpenAI models
LLMs take a prompt as input and optionally a system prompt or context and
return a response in the form of a string (excluding original prompt or special tokens).

In the future, we also expect support for other providers and models.

A dedicated LLMRouter can be used to instantiate the corresponding LLMs.
"""
from .llms import BaseLLM, HuggingFaceLLM, LLMRouter, OpenAILLM

all = ["BaseLLM", "HuggingFaceLLM", "OpenAILLM", "LLMRouter"]
