"""Helpers for string manipulations, cleaning, comparison, etc"""
import re


def clean_and_extract_multiple_choice(input_string):
    """Clean response string using regex."""
    # Remove everything between [] and <> (including the brackets)
    cleaned_string = re.sub(r"\[.*?\]|\<.*?\>", "", input_string)
    cleaned_string = cleaned_string.replace("\r\n", "").replace("\n\n", "").lstrip()

    # Check if the string contains "Answer:"
    answer_match = re.search(r"\bAnswer:\s*([A-D])\.", cleaned_string.strip())

    if answer_match:
        # If "Answer:" is found, return the letter
        return answer_match.group(1)

    # If no "Answer:" is found, extract the first valid letter
    match = re.search(r"\b([A-D])\b", cleaned_string)

    if match:
        return match.group(1)

    return cleaned_string


def clean_and_extract_open_text_answers(input_string):
    """Clean response string using regex."""
    # Remove everything between [] and <> (including the brackets)
    cleaned_string = re.sub(r"\[.*?\]|\<.*?\>", "", input_string)
    cleaned_string = cleaned_string.replace("\r\n", "").replace("\n\n", "").lstrip()
    return cleaned_string
