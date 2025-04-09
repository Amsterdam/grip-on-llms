"""Mapping of model names to full HF model ids as well as adding necessary parameters."""


TRANSLATOR_MAPPING = {
    "nllb-1b": {
        "id": "facebook/nllb-200-distilled-1.3B",
        "kwargs": {},
        "get_lang_code": {"NL": "nld_Latn", "EN": "eng_Latn"},
    },
    "mbart-large": {
        "id": "facebook/mbart-large-50-many-to-many-mmt",
        "kwargs": {},
        "get_lang_code": {"NL": "nl_XX", "EN": "en_XX"},
    },
}
