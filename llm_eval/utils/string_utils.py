"""Helpers for string manipulations, cleaning, comparison, etc"""
import re
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


def clean_and_extract_multiple_choice(input_string):  # noqa
    """
    Parse multiple choice response using comprehensive regex patterns.

    Uses sophisticated regex patterns to handle various response formats:
    - Single letters: "A", "B", "C", "D"
    - With punctuation: "A)", "B.", "C:"
    - With prefixes: "Answer A", "Option B", "Keuze C"
    - In sentences: "The answer is A", "Het antwoord is B"
    - Case insensitive matching
    - Dutch language patterns

    Args:
        input_string (str): The model's response text

    Returns:
        str: The extracted choice label or cleaned string if no valid choice
    """
    if not input_string:
        return "INVALID"

    valid_choices = ["A", "B", "C", "D", "E"]

    # Remove everything between [] and <> (including the brackets)
    response_clean = re.sub(r"\[.*?\]|\<.*?\>", "", input_string)
    response_clean = response_clean.replace("\r\n", "").replace("\n\n", "").strip()

    # Regex patterns for extracting choice labels (in order of specificity)
    patterns = _get_choice_patterns()

    # Try patterns in order of priority
    for pattern in patterns:
        matches = re.finditer(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            label = match.group(1).upper()
            if label in valid_choices:
                return label

    # Special handling for Dutch responses
    dutch_patterns = [
        (r"\b(?:eerste|1e)\b.*\b(?:optie|keuze)\b", "A"),
        (r"\b(?:tweede|2e)\b.*\b(?:optie|keuze)\b", "B"),
        (r"\b(?:derde|3e)\b.*\b(?:optie|keuze)\b", "C"),
        (r"\b(?:vierde|4e)\b.*\b(?:optie|keuze)\b", "D"),
    ]

    for pattern, label in dutch_patterns:
        if re.search(pattern, response_clean, re.IGNORECASE) and label in valid_choices:
            return label

    # Final fallback: look for any single letter that's a valid choice
    single_letters = re.findall(r"\b([A-Z])\b", response_clean.upper())
    for letter in single_letters:
        if letter in valid_choices:
            return letter
    # If no valid choice found
    return "INVALID"


def _get_choice_patterns():
    """Get regex patterns for choice extraction in order of priority."""
    return [
        # 1. Explicit answer formats - highest priority
        r"\b(?:antwoord|answer|keuze|choice|optie|option)\s*(?:is\s*)?([A-D])\b",
        r"\b(?:ik\s+kies\s+(?:voor\s+)?|i\s+choose\s+)([A-D])\b",
        r"\b(?:het\s+(?:juiste\s+)?antwoord\s+is\s+)([A-D])\b",
        r"\b(?:the\s+(?:correct\s+)?answer\s+is\s+)([A-D])\b",
        # 2. Answer: format - high priority
        r"\bAnswer:\s*([A-D])[\.\)\,\;]?\s*(?:\s|$)",
        # 3. Single letter with common punctuation - high priority
        r"\b([A-D])[\)\.\:\,\;]\s*(?:\s|$)",
        # 4. Single letter at start of line/response - medium priority
        r"^([A-D])\b",
        r"\n([A-D])\b",
        # 5. Single letter with word boundaries - lower priority
        r"\b([A-D])\b",
    ]


def clean_and_extract_open_text_answers(input_string):
    """Clean response string using regex."""
    # Remove everything between [] and <> (including the brackets)
    cleaned_string = re.sub(r"\[.*?\]|\<.*?\>", "", input_string)
    cleaned_string = cleaned_string.replace("\r\n", "").replace("\n\n", "").lstrip()
    return cleaned_string
