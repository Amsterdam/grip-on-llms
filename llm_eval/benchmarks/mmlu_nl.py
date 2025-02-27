"""
The MMLU benchmark ....

References:
Hendrycks, Dan, et al. "Measuring massive multitask language understanding." arXiv preprint arXiv:2009.03300 (2020).
"""
import json
from pathlib import Path
from pprint import pprint
import requests

from llm_eval.benchmarks.base import BaseBenchmark


class MMLU_NL(BaseBenchmark):
    """
    """
    def __init__(self, source_url, data_dir):
        self.name = "MMLU-NL"
        self.source_url = source_url
        self.data_dir = Path(data_dir) / self.name
        self.data_path = self.data_dir / "data.json"
        self.data = None

        self._prep_data()

    def _prep_data(self):
        """Download the benchmark data if not available and load it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_path.exists():
            self._download_data()
        self._load_data()

    def _download_data(self):
        """Download the data"""
        response = requests.get(self.source)
        with open(self.data_path, "wb") as f:
            f.write(response.content)

    def _load_data(self):
        self.data = json.load(open(self.data_path, "rb"))
        # pprint(self.data)
 
    def run(self, llm):
        """Run the MMLU benchmark using the provided LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is not loaded.")
        
        results = []
        for entry in self.data:
            prompt = (
                "Answer the following question. Only state A, B, C, D"
                f"{entry['instruction']}\n"
                f"A. {entry['option_a']}\n"
                f"B. {entry['option_b']}\n"
                f"C. {entry['option_c']}\n"
                f"D. {entry['option_d']}\n"
                "Answer:"
            )
            # pprint(prompt)

            expected_answer = entry["answer"]
            llm_response = llm.prompt(prompt)
            result = {
                "prompt": prompt,
                "expected": expected_answer,
                "response": llm_response,
                "correct": llm_response.strip().lower() == expected_answer.strip().lower()
            }
            results.append(result)
            if len(results) > 2:
                break

        pprint(results)

        return results
