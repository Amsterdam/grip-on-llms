"""
Department: Juridisch Bureau
Team: Data Science & AI - Juridisch Bureau
Application: KOG Classifier
Description:
The teams wants to classify legal cases whether they are "Kennelijk Ongegrond", meaning that
there are no valid arguments for winning the case.

Metrics:
1. Accuracy
2. Precision / Recall
3. F1-score
4. Confusion Matrix
"""

from typing import List
import pandas as pd
from pathlib import Path

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.schemas import BenchmarkEvaluation, RunItem
from llm_eval.utils.setup_utils import benchmark_data_folder


class KOGClassifier(BaseBenchmark):
    """
    KOG Classifier benchmark for measuring performance on recognizing "Kennelijk Ongegrond" legal cases.
    """

    def __init__(
        self,
        benchmark_name="kog_classifier",
        source_url=None,
        data_path=Path(benchmark_data_folder) / "KOG" / "Dataset.xlsx",
        prompt_path=Path(benchmark_data_folder) / "KOG" / "prompt.txt",
        hf_repository=None,
        preferred_response_format="open_text",
    ):
        """Initialize KOG Classifier benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            source_url=source_url,
            data_path=data_path,
            hf_repository=hf_repository,
            preferred_response_format=preferred_response_format,
        )
        self.prompt_path = prompt_path
        # Load data when benchmark is initialized
        self.data = self._load_data()
        self.prompt = self._load_prompt()

    def _load_data(self):
        """Load data from local file."""
        df = pd.read_excel(self.data_path)
        return df.to_dict('records')

    def _get_hashing_data_for_sampling(self):
        """Get data for consistent sampling."""
        # TODO: Implement based on data structure
        # Usually return a list of unique identifiers
        return [entry["id"] for entry in self.data]

    def _run_task(self, llm, n_samples=0) -> List[RunItem]:
        """Run the classification task using the provided LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is not loaded.")

        prompts = []
        for entry in self.data:
            prompt = self.prompt.format(
                bezwaar=entry['adviestekst']
            )
            prompts.append(prompt)

        # Get LLM responses
        responses = llm.process_batch(
            prompts, 
            response_format=self.preferred_response_format
        )

        # Create RunItem objects
        run_items = []
        for i, (entry, response) in enumerate(zip(self.data, responses)):
            target = 1 if entry.get("Beoordeling Loubna") == "KOG" else 0
            predicted = self._parse_answer(response.processed_response)
            
            # Correctness evaluation: only consider valid predictions
            is_correct = (predicted != "INVALID") and (predicted == target)

            run_item = RunItem(
                **response.model_dump(),
                prompt=prompts[i],
                prompt_idx_original=i,
                target=target,
                predicted=predicted,
                correct=is_correct,
            )
            run_items.append(run_item)
        return run_items

    def _parse_answer(self, answer):
        """Parse the answer from KOG."""
        if not answer or not isinstance(answer, str):
            return "INVALID"
            
        try:
            # Split on dash and take the first part as label
            parts = answer.split("-", 1)
            label = parts[0].strip().lower()
            if "niet kog" in label:
                return 0
            elif "kog" in label and "niet kog" not in label:
                return 1
            else:
                return "INVALID"
        except Exception:
            return "INVALID"


    def _calculate_metrics(self, run_output: List[RunItem]) -> BenchmarkEvaluation:
        """Calculate comprehensive evaluation metrics."""
        
        # Separate valid and invalid predictions
        valid_items = [item for item in run_output if item.predicted != "INVALID"]
        invalid_items = [item for item in run_output if item.predicted == "INVALID"]
        
        n_total = len(run_output)
        n_valid = len(valid_items)
        n_invalid = len(invalid_items)
        
        # Basic counts
        n_correct = sum(1 for item in run_output if item.correct)
        
        # Overall accuracy (including invalid as incorrect)
        overall_accuracy = n_correct / n_total if n_total > 0 else 0.0
        
        # Valid-only accuracy (accuracy on items with valid predictions)
        valid_accuracy = (sum(1 for item in valid_items if item.correct) / n_valid) if n_valid > 0 else 0.0
        
        # Initialize metrics dictionary
        metrics = {
            "accuracy": overall_accuracy,
            "valid_accuracy": valid_accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "invalid_rate": n_invalid / n_total if n_total > 0 else 0.0,
        }
        
        # Calculate precision, recall, F1 only on valid predictions
        if n_valid > 0:
            # Extract targets and predictions for valid items only
            targets = [item.target for item in valid_items]
            predictions = [item.predicted for item in valid_items]
            
            # Calculate confusion matrix components
            tp = sum(1 for t, p in zip(targets, predictions) if t == 1 and p == 1)  # True Positives
            tn = sum(1 for t, p in zip(targets, predictions) if t == 0 and p == 0)  # True Negatives
            fp = sum(1 for t, p in zip(targets, predictions) if t == 0 and p == 1)  # False Positives
            fn = sum(1 for t, p in zip(targets, predictions) if t == 1 and p == 0)  # False Negatives
            
            # Store confusion matrix
            confusion_matrix =  {
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn
            }
            metrics["confusion_matrix"] = confusion_matrix
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.update({
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            })
        else:
            # No valid predictions - set all metrics to 0
            metrics.update({
                "confusion_matrix": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "specificity": 0.0,
                "sensitivity": 0.0,
                "precision_kog": 0.0,
                "recall_kog": 0.0,
                "precision_niet_kog": 0.0,
                "recall_niet_kog": 0.0,
            })

        return BenchmarkEvaluation(
            metrics=metrics,
            total_samples=len(run_output),
        )


    def _get_own_metadata(self):
        """Get benchmark-specific metadata."""
        # TODO: Add relevant metadata with team
        metadata = {
            "task_type": "classification",
            "language": "dutch",
        }
        return metadata

    def _load_prompt(self):
        """Load prompt template from text file."""
        try:
            with open(self.prompt_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read().strip()
            return prompt_template
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {self.prompt_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt from {self.prompt_path}: {str(e)}")