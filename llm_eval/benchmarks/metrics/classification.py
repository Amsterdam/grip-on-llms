"""Implementation of (tiny) classification metrics"""

import numpy as np
import tinyBenchmarks as tb


def tiny_scores(predictions, task):
    """Given results (list of correct/incorrect calculate irt, pirt & gpirt"""
    padded_to_100 = np.array(predictions + [0] * (100 - len(predictions)))
    accuracy = tb.evaluate(padded_to_100, task)
    return accuracy


if __name__ == "__main__":
    # Test padding
    benchmark = "truthfulqa"
    # benchmark = "mmlu"
    for x in range(0, 101):
        print(f"----- {x} -----")
        predictions = [1] * x
        print(f"No pad:       {tiny_scores(predictions, benchmark)}")
        predictions = [1] * x + [0] * (100 - x)
        print(f"1s + 0s:      {tiny_scores(predictions, benchmark)}")
        predictions = [True] * x + [False] * (100 - x)
        print(f"True + False: {tiny_scores(predictions, benchmark)}")
        predictions = [1] * x + [-1] * (100 - x)
        print(f"1s + -1s:     {tiny_scores(predictions, benchmark)}")
