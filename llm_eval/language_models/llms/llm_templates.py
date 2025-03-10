"""
Dealing with (huggingface) model templating.
Currently, we are using a set of own basic templates for more transparency and control.
The aim is to transfer to the hugging face chat templating wherever possible and convenient.
More information: https://huggingface.co/docs/transformers/main/chat_templating
"""
import re

DEFAULT_INTRO = (
    # Default alpaca-style instruction added when no context
    "Below is an instruction that describes a task. "
    + "Write a response that appropriately completes the request."
)
DEFAULT_INTRO_CONTEXT = (
    # Default alpaca-style instruction added when provided context
    "Below is an instruction that describes a task, "
    + "paired with an input that provides further cotext. "
    + "Write a response that appropriately completes the request."
)
USING_CONTEXT = "Using this information: {context} " + "answer the Question: {prompt}"


def falcon_prompt(prompt, context=None, system=""):
    if context:
        instruction = USING_CONTEXT.format(context=context, prompt=prompt)
        formatted_prompt = f"{system} {instruction}"
    else:
        formatted_prompt = f"{system} {prompt}"
    return formatted_prompt


def falcon_prompt_dutch(prompt, context=None, system=""):
    formatted_prompt = f"Vraag: {prompt}\nAntwoord:"
    return formatted_prompt


def llama_prompt(prompt, context=None, system=""):
    if context:
        system_msg = f"<<SYS>> {DEFAULT_INTRO_CONTEXT} {system} <</SYS>>"
        instruction = USING_CONTEXT.format(context=context, prompt=prompt)
        formatted_prompt = f"<s>[INST] {system_msg} {instruction} [/INST]"
    else:
        system_msg = f"<<SYS>> {DEFAULT_INTRO} {system} <</SYS>>"
        formatted_prompt = f"<s>[INST] {system_msg} {prompt} [/INST]"
    return formatted_prompt


def default_prompt(prompt, context=None, system=""):
    if context:
        formatted_prompt = USING_CONTEXT.format(context=context, prompt=prompt)
    else:
        formatted_prompt = prompt

    if system:
        formatted_prompt = f"{system}\n{formatted_prompt}"
    return formatted_prompt


template_mapping = {
    "falcon-7b-instruct": falcon_prompt,
    "falcon-40b-instruct": falcon_prompt,
    "falcon-7b-instruct-dutch": falcon_prompt_dutch,
    "falcon-40b-instruct-dutch": falcon_prompt_dutch,
    "mistral-7b-instruct": llama_prompt,
    "mixtral-7b-instruct": llama_prompt,
    "llama-7b": llama_prompt,
    "llama-7b-chat": llama_prompt,
    "llama-13b-chat": llama_prompt,
    "llama-70b-chat": llama_prompt,
    "tiny-llama": llama_prompt,
}


def format_prompt(prompt, model_name, context=None, system=""):
    template_function = template_mapping.get(model_name, default_prompt)
    formatted_prompt = template_function(prompt, context, system)
    formatted_prompt = re.sub(r"[ \t]+", " ", formatted_prompt).strip()
    return formatted_prompt


def get_default_template(model_name, context=False, system=""):
    context_msg = "{context}" if context else ""
    system_msg = "{system}" if system else ""
    return format_prompt(
        prompt="{prompt}", model_name=model_name, context=context_msg, system=system_msg
    )
