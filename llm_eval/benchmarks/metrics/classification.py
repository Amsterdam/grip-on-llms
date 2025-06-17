"""Implementation of (tiny) classification metrics"""

import numpy as np
import tinyBenchmarks as tb


def tiny_scores(predictions, task):
    """Given results (list of correct/incorrect calculate irt, pirt & gpirt"""
    padded_to_100 = np.array(predictions + [0] * (100 - len(predictions)))
    accuracy = tb.evaluate(padded_to_100, task)
    return accuracy
