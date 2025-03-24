import json
from pathlib import Path
import requests
from tqdm import tqdm

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm_eval.benchmarks.base import BaseBenchmark

# Prompt template in Dutch for summarizing a news article
prompt_template = (
    "Hieronder staat een nieuwsartikel.\n"
    "Vat het artikel samen in ongeveer 50 woorden.\n"
    "Artikel: {article}\n"
    "Samenvatting:"
)

class Summarization(BaseBenchmark):
    """
    De Summarization benchmark verwacht een bron JSON-bestand met taken zoals:
    {
         "article": "De volledige tekst van het nieuwsartikel...",
         "reference": "De menselijke samenvatting in het Nederlands.",
         "id": "nieuws/voorbeeld/0"
    }

    Deze benchmark meet de kwaliteit van automatisch gegenereerde samenvattingen
    door ze later te vergelijken met de referenties, bijvoorbeeld met behulp van ROUGE.
    """

    def __init__(self, source_url, data_dir, categories=None):
        self.name = "Summarization-NL"
        self.source_url = source_url
        self.data_dir = Path(data_dir) / self.name
        self.data_path = self.data_dir / "data.json"
        self.categories = categories
        self.data = None
        self.translation_model = None
        self.results = {}
        self._prep_data()

    def _prep_data(self):
        """Download de benchmark data indien niet beschikbaar en laad het in."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_path.exists():
            self._download_data()
        self._load_data()

    def _download_data(self):
        """Download de data van de opgegeven URL."""
        response = requests.get(self.source_url)
        with open(self.data_path, "wb") as f:
            f.write(response.content)

    def _load_data(self):
        """Laad de data vanuit het JSON-bestand."""
        self.data = json.load(open(self.data_path, "rb"))
        if self.categories:
            self.data = [
                entry for entry in self.data
                if any(cat in entry["id"] for cat in self.categories)
            ]

    def _run_task(self, llm, results_path=None):
        """Voer de Summarization benchmark uit met de gegeven LLM."""
        if self.data is None:
            raise ValueError("Benchmark data is niet geladen.")
        benchmark_results = []
        for entry in tqdm(self.data):
            prompt = prompt_template.format(
                article=entry["article"]
            )
            expected_summary = entry["reference"]
            llm_response = llm.prompt(prompt)
            result = {
                "prompt": prompt,
                "reference": expected_summary,
                "response": llm_response,
            }
            benchmark_results.append(result)
        return benchmark_results

    def _calculate_metric(self, results=None):
        """
        Bereken een eenvoudige evaluatiemaatstaf voor samenvatting.
        Dit voorbeeld gebruikt een dummy metric gebaseerd op woordoverlap
        tussen de referentie en de gegenereerde samenvatting.
        """
        if not results:
            return 0.0
        total_score = 0
        for entry in results:
            ref_words = set(entry["reference"].split())
            resp_words = set(entry["response"].split())
            if ref_words:
                total_score += len(ref_words.intersection(resp_words)) / len(ref_words)
        return total_score / len(results)

    @staticmethod
    def translate_text(text, model, tokenizer):
        """
        Placeholder vertaalfunctie.
        Vervang deze met een API-call of model (bijv. Google Translate, MarianMT, etc.)
        die de tekst naar de gewenste taal vertaalt.
        """
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                                           max_length=512)
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    @classmethod
    def translate_and_upload_benchmarks(cls):
        """
        Downloadt de XSUM en CNN/DailyMail benchmarks van Huggingface,
        vertaalt de relevante velden naar het Nederlands, en pusht
        de nieuwe datasets terug naar Huggingface.

        Voor XSUM:
            - Input velden: "document" (artikel) en "summary" (referentie)
        Voor CNN/DailyMail:
            - Input velden: "article" (artikel) en "highlights" (referentie)

        De nieuwe datasets worden opgeslagen met de velden:
            "article": vertaald artikel
            "reference": vertaald referentie
        """
        # Define datasets and corresponding field mappings
        datasets_to_translate = {
            "EdinburghNLP/xsum": {"article": "document", "reference": "summary", "repo_name": "Dutch_xsum"},
            "abisee/cnn_dailymail": {"article": "article", "reference": "highlights", "repo_name": "Dutch_cnn_dailymail"}
        }


        model_name = "facebook/nllb-200-distilled-1.3B"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for hf_dataset, mapping in datasets_to_translate.items():
            print(f"Processing dataset: {hf_dataset}")
            # Load the dataset (here we load the 'test' split as an example)
            ds = load_dataset(hf_dataset)['test']
            # Process each example and translate the required fields
            def translate_example(example):
                # Extract source fields using the provided mapping
                article = example.get(mapping["article"], "")
                reference = example.get(mapping["reference"], "")
                # Translate text to Dutch (placeholder; replace with real translation)
                translated_article = cls.translate_text(article, model, tokenizer)
                translated_reference = cls.translate_text(reference, model, tokenizer)
                return {"article": translated_article, "reference": translated_reference}

            # Apply translation on the dataset
            translated_ds = ds.map(translate_example, remove_columns=ds.column_names)

            # Optionally, add an "id" field if needed
            def add_id(example, idx):
                example["id"] = f"{mapping['repo_name']}/{idx}"
                return example

            translated_ds = translated_ds.map(add_id, with_indices=True)

            # Push the new dataset to Huggingface Hub
            print(f"Pushing translated dataset to Huggingface Hub as repo: {mapping['repo_name']}")
            translated_ds.push_to_hub(mapping["repo_name"], private=False)

        print("Vertaling en upload van benchmarks voltooid.")

if __name__ == "__main__":
    Summarization.translate_and_upload_benchmarks()
    # test = Summarization("test", "test")