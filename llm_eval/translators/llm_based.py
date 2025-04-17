"""Support for LLM-based translators"""
from llm_eval.translators.base import BaseTranslator

default_prompt = (
    "{EXTRA_INSTRUCTIONS}\n"
    "Translate the following text from {SOURCE_LANG} to {TARGET_LANG}.\n"
    "Text: {TEXT}\n"
    "Translation:"
)


class LLMTranslator(BaseTranslator):
    """
    A class to support llm-based translations.
    Expects an LLM instance which implements a prompt() function.
    """

    def __init__(self, model_name, source_lang, target_lang, llm, params=dict):
        super().__init__(
            model_name, source_lang, target_lang, params if params is not None else {}
        )

        self.llm = llm

    def _translate(
        self,
        text,
        source_lang=None,
        target_lang=None,
        max_length=512,
        use_default_prompt=True,
        **kwargs
    ):
        """Translte text"""
        source_lang = source_lang if source_lang else self.source_lang
        target_lang = target_lang if target_lang else self.target_lang

        if use_default_prompt:
            prompt = default_prompt.format(
                EXTRA_INSTRUCTIONS="",
                SOURCE_LANG=source_lang,
                TARGET_LANG=target_lang,
                TEXT=text,
            )
        else:
            prompt = text

        translation = self.llm.prompt(prompt)

        return translation
