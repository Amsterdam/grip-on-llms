"""
Dealing with (huggingface) model templating.
Currently, we are using a set of own basic templates for more transparency and control.
The aim is to transfer to the hugging face chat templating wherever possible and convenient.
More information: https://huggingface.co/docs/transformers/main/chat_templating
Disclaimer: Prompts using context still need to be thoroughly tested.
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


def llama_prompt(prompt, context=None, system=""):
    if context:
        system_msg = f"<<SYS>> {DEFAULT_INTRO_CONTEXT} {system} <</SYS>>"
        instruction = USING_CONTEXT.format(context=context, prompt=prompt)
        formatted_prompt = f"<s>[INST] {system_msg} {instruction} [/INST]"
    else:
        system_msg = f"<<SYS>> {DEFAULT_INTRO} {system} <</SYS>>"
        formatted_prompt = f"<s>[INST] {system_msg} {prompt} [/INST]"
    return formatted_prompt


def mistral_prompt(prompt, context=None, system=""):
    system_msg = f" {system}\n\n" if system else ""
    if context:
        instruction = USING_CONTEXT.format(context=context, prompt=prompt)
    else:
        instruction = prompt

    formatted_prompt = f"<s>[INST] {system_msg}{instruction}[/INST]"
    return formatted_prompt


def system_user_assistant_prompt(prompt, context=None, system=""):
    system_msg = f"<|system|>\n{system}\n" if system else ""

    if context:
        instruction = USING_CONTEXT.format(context=context, prompt=prompt)
    else:
        instruction = prompt
    formatted_prompt = f"{system_msg}<|user|>\n{instruction}\n<|assistant|>\n"
    return formatted_prompt


def falcon3_prompt(prompt, context=None, system=""):
    default_falcon_system = (
        "You are a helpful, friendly assistant Falcon3 from TII, "
        "try to follow instructions as much as possible."
    )
    system = system if system else default_falcon_system
    formatted_prompt = system_user_assistant_prompt(prompt, context, system)
    return formatted_prompt


def phi_prompt(prompt, context=None, system=""):
    formatted_prompt = (
        "<|im_start|>system<|im_sep|>"
        f"{DEFAULT_INTRO}"
        "<|im_start|>user<|im_sep|>"
        f"{prompt}<|im_end|>"
    )
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
    "falcon3-7b-instruct": falcon3_prompt,
    "mistral-7b-instruct-v0.3": mistral_prompt,
    "tiny-llama": system_user_assistant_prompt,
    "phi-4-mini-instruct": phi_prompt,
    "llama-3.2-3b-instruct": llama_prompt,
    "llama-3.1-8b-instruct": llama_prompt,
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


if __name__ == "__main__":
    for model in template_mapping.keys():
        print(f"----- {model} -----")
        print(template_mapping[model](prompt="PROMPT", context=None, system=None))
