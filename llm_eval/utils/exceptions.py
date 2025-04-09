"""Custom exceptions shared accross the whole llm_eval"""


class EmptyResponseError(Exception):
    """Exception raised when the LLM returns an empty response."""

    def __init__(self, message="The LLM returned an empty response."):
        self.message = message
        super().__init__(self.message)
