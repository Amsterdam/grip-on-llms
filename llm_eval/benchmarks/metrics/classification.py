"""Implementation of (tiny) classification metrics"""

import numpy as np
import tinyBenchmarks as tb

ANSWERS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}

ANSWERS_REVERSE = {v: k for k, v in ANSWERS.items()}


def tiny_accuracy(predictions, task):
    """Given results, calculate desired score"""
    response_ids = np.array(
        [ANSWERS_REVERSE.get(pred, -1) for pred in predictions] + [-1] * (100 - len(predictions))
    )
    accuracy = tb.evaluate(response_ids, task)
    return accuracy
