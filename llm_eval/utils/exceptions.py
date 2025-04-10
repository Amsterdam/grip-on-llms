"""Custom exceptions shared accross the whole llm_eval"""


class EmptyResponseError(Exception):
    """Exception raised when the LLM returns an empty response."""

    def __init__(self, message="The LLM returned an empty response."):
        self.message = message
        super().__init__(self.message)


class UnsupportedModelError(Exception):
    """Exception raised for unsupported models."""

    def __init__(self, model_name=None, supported_models=None):
        self.model_name = model_name
        self.supported_models = supported_models
        super().__init__(self._generate_message())

    def _generate_message(self):
        error_msg = (
            f"Unsupported Model {self.model_name}" if self.model_name else "Unsupported Model.\n"
        )
        if self.supported_models:
            error_msg += f"Choose from: {self.supported_models}"
        return error_msg


class TranslatorMissingError(ValueError):
    """Exception raised when a translator is missing but a language is specified."""

    def __init__(self, language=None):
        self.language = language
        super().__init__(self._generate_message())

    def _generate_message(self):
        error_msg = "Missing a necessary translator.\n"
        if self.language:
            error_msg += f"Ensure {self.language} support"
        return error_msg
