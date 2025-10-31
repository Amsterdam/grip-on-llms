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

import re
from pathlib import Path
from typing import List

import pandas as pd

from llm_eval.benchmarks.base import BaseBenchmark
from llm_eval.utils.schemas import BenchmarkEvaluation, RunItem
from llm_eval.utils.setup_utils import benchmark_data_folder

PROMPT_TEMPLATE = (
    "Jij bent een juridisch expert die bezwaarschriften beoordeelt. Je taak is om "
    'objectief vast te stellen of een bezwaar onder "Kennelijk Ongegrond" (KOG) valt '
    "of niet (niet-KOG).\n"
    "Geef je antwoord in exact één regel: begin met 'KOG –' of 'niet-KOG –', gevolgd "
    "door een korte reden in het Nederlands.\n"
    "Gebruik géén formele conclusies, géén juridische afwijzingen, géén paragrafen of "
    "opsommingen, en géén andere labels.\n"
    "Let op:\n"
    "- Beoordeel alléén wat letterlijk in het bezwaar staat.\n"
    "- Voeg géén aannames toe.\n"
    "- Bezwaren kunnen in het Nederlands of in het Engels zijn geschreven.\n"
    "- Wees strikt in je beoordeling en volg exact de regels hieronder.\n"
    "Geef alleen KOG als de bezwaarmaker de overtreding erkent én een geldige KOG-reden "
    "noemt, zonder dat er ook een niet-KOG-reden in hetzelfde bezwaar staat.\n"
    "Zodra er ook een niet-KOG-reden genoemd wordt, is het altijd niet-KOG.\n"
    "Een bezwaar is alleen KOG als:\n"
    "De bezwaarmaker impliciet of expliciet erkent dat hij of zij de overtreding heeft "
    "begaan, én daarbij een reden noemt.\n"
    "De redenen zijn:\n"
    "-De container was eenmalig of soms vol of kapot.\n"
    "-Onbekendheid met de regels of niet weten dat iets niet mocht (dit telt als "
    "impliciete erkenning van de overtreding)\n"
    "- Afval is eerder aangeboden vanwege vakantie.\n"
    "- Er wordt gevraagd om coulance door middel van een waarschuwing\n"
    "- De bezwaarmaker aangeeft dat het afval mogelijk eruit is gevallen of door een "
    "ander uit de container is gehaald.\n"
    "- De bezwaarmaker is milieubewust.\n"
    "- De bezwaarmaker zegt normaal altijd afval goed weg te gooien.\n"
    "- De boete kan niet worden betaald.\n"
    "- Slechts één poststuk is ontvangen.\n"
    "- Er waren geen andere containers in de straat.\n\n"
    "Een bezwaar is niet-KOG als:\n"
    "De bezwaarmaker de overtreding ontkent. Er wordt aangegeven dat hij/zij het niet "
    "heeft gedaan of nooit heeft kunnen doen.\n"
    "Hierbij kunnen ook de volgende redenen worden genoemd:\n"
    "- De bezwaarmaker vraagt om bewijs.\n"
    "- De bezwaarmaker begrijpt niet waarom hij/zij de boete heeft ontvangen.\n"
    "- Structurele overlast: alleen wanneer containers altijd vol of kapot zijn, of "
    "een terugkerend probleem in de straat. Tijdelijke, eenmalige of uitzonderlijke "
    "situaties, zoals afgesloten containers door bouw of een eenmalige stapel afval, zijn "
    "**geen** structurele overlast en tellen niet als niet-KOG.\n"
    "- De bezwaarmaker benoemt structurele ongelijkheid (bijvoorbeeld: alle buren "
    "hebben hetzelfde gedaan, maar alleen hij/zij krijgt een boete).\n"
    "- De bezwaarmaker geeft aan medische beperkingen te hebben.\n"
    "- De bezwaarmaker geeft aan contact te hebben gehad met de gemeente of een "
    "melding te hebben gemaakt.\n"
    "- De dader is minderjarig (tot en met 16 jaar)\n"
    "- De bezwaarmaker schuift de schuld af op een ander (bijv. huisgenoot, kind of "
    "buren), zonder zelf enige verantwoordelijkheid te erkennen.\n"
    "- De bezwaarmaker stuurt bewijzen mee die aantonen dat hij/zij het niet kan zijn "
    "geweest (bijvoorbeeld vliegtickets).\n"
    "- De bezwaarmaker zegt het afval niet te herkennen of vermoedt dat iemand anders "
    "het met zijn/haar naam erin heeft weggegooid.\n\n"
    "Voorbeelden:\n"
    "Voorbeeld 1 (KOG)  \n"
    'Bezwaar: "De containers zaten vol, ik kon mijn vuil niet elders kwijt er waren '
    'geen andere containers in de buurt."  \n'
    "Uitkomst: KOG - De bezwaarmaker erkent dat hij afval naast een (eenmalige) volle "
    "container heeft geplaatst.\n"
    "Voorbeeld 2 (niet-KOG)  \n"
    'Bezwaar: "Ik was niet in de buurt en weet van niets."  \n'
    "Uitkomst: niet-KOG - De bezwaarmaker ontkent betrokkenheid bij de overtreding. "
    "Hierom is het niet-KOG.\n"
    "Voorbeeld 3 (niet-KOG)  \n"
    'Bezwaar: "Ik wil bewijs zien, ik snap niet waarom ik deze boete krijg."\n'
    "Uitkomst: niet-KOG - Er is onbegrip en geen erkenning van de overtreding. Hierom "
    "is het niet-KOG.\n"
    "Voorbeeld 4 (KOG)  \n"
    'Bezwaar: "Ik was op vakantie en zette mijn afval eerder buiten."  \n'
    "Uitkomst: KOG - De bezwaarmaker erkent de overtreding. Hierom is het KOG.\n"
    "Voorbeeld 5 (niet-KOG)  \n"
    'Bezwaar: "Mijn huisgenoot heeft het afval weggegooid en onjuist aangeboden."  \n'
    "Uitkomst: niet-KOG - De bezwaarmaker erkent geen eigen verantwoordelijkheid en "
    "erkent geen eigen verantwoordelijkheid. Hierom is het niet-KOG\n"
    "Voorbeeld 6 (KOG)  \n"
    'Bezwaar: "Ik ben milieubewust en gooi normaal alles altijd netjes weg. Dit is '
    "deze keer niet gebeurd door de overvolle container waardoor ik het ernaast heb "
    'gelegd."  \n'
    "Uitkomst: KOG - De bezwaarmaker erkent de overtreding en benoemt geen structurele "
    "volle container. Hierom is het KOG\n"
    "Voorbeeld 7 (niet-KOG)  \n"
    'Bezwaar: "Ik heb mijn dochter van 15 gevraagd om het afval weg te gooien, zij '
    'wist niet dat dit niet mocht."  \n'
    "Uitkomst: niet-KOG - De bezwaarmaker schuift de schuld af op een minderjarige, "
    "omdat het om een minderjarige gaat is het niet-KOG.\n"
    "Voorbeeld 8 (KOG)  \n"
    'Bezwaar: "Ik heb het karton in de container gedaan, maar iemand heeft het uit de '
    'container gehaald en ernaast gelegd."\n'
    "Uitkomst: KOG - De bezwaarmaker erkent dat hij het karton zelf heeft weggegooid, "
    "ook al is het er later uitgehaald door een ander.\n"
    "Voorbeeld 9 (niet-KOG)\n"
    'Bezwaar: "Ik kan het nooit zijn geweest want ik was op dat moment in het '
    'buitenland."  \n'
    "Uitkomst: niet-KOG - De bezwaarmaker ontkent betrokkenheid bij de overtreding en "
    "geeft aan in het buitenland te zijn geweest. Hierom is het niet-KOG.\n"
    "Voorbeeld 10 (niet-KOG)\n"
    'Bezwaar: "Ik heb medische klachten (hersenaandoening/hernia) waardoor ik uit '
    'noodzaak het afval verkeerd heb aangeboden."  \n'
    "Uitkomst: niet-KOG - De bezwaarmaker geeft medische klachten als argument voor het "
    "verkeerd aanbieden van afval. Hierom is het niet-KOG.\n"
    "Voorbeeld 11 (niet-KOG)\n"
    'Bezwaar: "Ik denk dat iemand anders iets met mijn naam erop heeft weggegooid, ik '
    'heb dit niet gedaan."\n'
    "Uitkomst: niet-KOG - De bezwaarmaker ontkent de overtreding en schuift de schuld "
    "af.\n"
    "Voorbeeld 12 (KOG)\n"
    'Bezwaar: "Ik heb mijn afval netjes weggegooid in de container, alleen is het dan '
    'waarschijnlijk eruit gevallen."\n'
    "Uitkomst: KOG - De bezwaarmaker geeft aan dat het afval waarschijnlijk eruit is "
    "gevallen, wat geen geldige reden is en het bezwaar hierdoor KOG is.\n"
    "Wees consistent: geef per bezwaar exact één regel terug met het label 'KOG –' of "
    "'niet-KOG –' en een korte reden.\n"
    "Bezwaarschrift:\n"
    "{bezwaar}\n"
)


class KOGClassifier(BaseBenchmark):
    """
    KOG Classifier benchmark for measuring performance on recognizing "Kennelijk Ongegrond"
    legal cases.
    """

    def __init__(
        self,
        benchmark_name="KOG-Classifier",
        source_url=None,
        data_path=benchmark_data_folder,
        hf_repository=None,
        preferred_response_format=None,
    ):
        """Initialize KOG Classifier benchmark."""
        super().__init__(
            benchmark_name=benchmark_name,
            source_url=source_url,
            hf_repository=hf_repository,
            preferred_response_format=preferred_response_format,
        )
        # Load data when benchmark is initialized
        self.data_path = Path(data_path) / benchmark_name / "Dataset.xlsx"
        self.data = self._load_data()

    def _load_data(self):
        """Load data from local file."""
        df = pd.read_excel(self.data_path)
        return df.to_dict("records")

    def _get_hashing_data_for_sampling(self):
        """Get data for consistent sampling."""
        # Usually return a list of unique identifiers
        return [entry["id"] for entry in self.data]

    def _run_task(self, llm, n_samples=0) -> List[RunItem]:
        """Run the classification task using the provided LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is not loaded.")

        prompts = []
        for entry in self.data:
            prompt = PROMPT_TEMPLATE.format(bezwaar=entry["Bezwaartekst"].strip())
            prompts.append(prompt)

        # Get LLM responses
        responses = llm.process_batch(prompts, response_format=self.preferred_response_format)

        # Create RunItem objects
        run_items = []
        for i, (entry, response) in enumerate(zip(self.data, responses)):
            target = 1 if entry.get("Beoordeling Loubna") == "KOG" else 0
            predicted = self._parse_answer(response.raw_response)

            # Correctness evaluation: only consider valid predictions
            is_correct = (predicted != "INVALID") and (predicted == target)

            run_item = RunItem(
                **response.model_dump(),
                prompt=prompts[i],
                prompt_idx_original=i,
                target=str(target),
                predicted=str(predicted),
                correct=is_correct,
            )
            run_items.append(run_item)
        return run_items

    def _parse_answer(self, response):
        """
        Simple parser for "KOG - {Explanation}" or "Not-KOG - {explanation}" format
        Returns: 1 (KOG), 0 (Not-KOG), or "invalid"
        """
        if not response:
            return "INVALID"

        # Clean and prepare response
        response = str(response).strip()

        # Check if it matches expected patterns (case-insensitive)
        if re.match(r"^KOG\s*[-–—]\s*.+", response, re.IGNORECASE):
            return 1
        elif re.match(r"^(Not|NIET|No)[\s-]*KOG\s*[-–—]\s*.+", response, re.IGNORECASE):
            return 0
        else:
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
        valid_accuracy = (
            (sum(1 for item in valid_items if item.correct) / n_valid) if n_valid > 0 else 0.0
        )

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
            targets = [int(item.target) for item in valid_items]
            predictions = [int(item.predicted) for item in valid_items]

            # Calculate confusion matrix components
            tp = sum(
                1 for t, p in zip(targets, predictions) if t == 1 and p == 1
            )  # True Positives
            tn = sum(
                1 for t, p in zip(targets, predictions) if t == 0 and p == 0
            )  # True Negatives
            fp = sum(
                1 for t, p in zip(targets, predictions) if t == 0 and p == 1
            )  # False Positives
            fn = sum(
                1 for t, p in zip(targets, predictions) if t == 1 and p == 0
            )  # False Negatives

            # Store confusion matrix
            confusion_matrix = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
            metrics["confusion_matrix"] = confusion_matrix
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                }
            )
        else:
            # No valid predictions - set all metrics to 0
            metrics.update(
                {
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
                }
            )

        return BenchmarkEvaluation(
            metrics=metrics,
            total_samples=len(run_output),
        )

    def _get_own_metadata(self):
        """Get benchmark-specific metadata."""
        metadata = {
            "task_type": "classification",
            "language": "dutch",
        }
        return metadata

    def _load_prompt(self):
        """Load prompt template from text file."""
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompt_template = f.read().strip()
            return prompt_template
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found at {self.prompt_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt from {self.prompt_path}: {str(e)}")
