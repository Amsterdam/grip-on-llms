"""Basic tests to check whether a benchmark works as expected"""
import logging

from env_setup import get_gpt_secrets, get_hf_secrets

from llm_eval.benchmarks import ARC, MMLU, AmsterdamSimplification, INTDuidelijkeTaal
from llm_eval.language_models import LLMRouter
from llm_eval.leaderboard import Leaderboard


def test_leaderboard():
    codecarbon_params = {
        "country_iso_code": "SWE",
        "region": "sweden",
        "allow_multiple_runs": True,
        "save_to_file": False,
        "pue": 1.185,
    }

    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        # "stop": None,
    }

    logging.info("Initializing GPT")
    gpt_secrets = get_gpt_secrets()
    gpt = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=gpt_secrets["API_ENDPOINT"],
        api_key=gpt_secrets["API_KEY"],
        api_version=gpt_secrets["API_VERSION"],
        params=gpt_params,
        uses_api=True,
    )

    hf_secrets = get_hf_secrets()
    hf_params = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.65,
        # "top_k": 25,
        "max_new_tokens": 200,
        "no_repeat_ngram_size": 3,
        "num_return_sequences": 1,
    }

    logging.info("Initializing some HF model")
    tinyllama = LLMRouter.get_model(
        provider="huggingface",
        model_name="tiny-llama",
        hf_token=hf_secrets["HF_TOKEN"],
        #        hf_cache=os.environ["HF_CACHE"],
        hf_cache=None,
        params=hf_params,
        uses_api=False,
    )

    logging.info("Setting up benchmarks")
    mmlu_source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_mmlu/nl_dev.json"
    data_folder = "./data"
    mmlu_nl_bench = MMLU("MMLU-NL", mmlu_source, data_folder, categories=["moral_disputes"])

    arc_source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_arc/nl_validation.json"
    data_folder = "./data"
    arc_nl_bench = ARC("ARC-NL", arc_source, data_folder, categories=["LEAP"])

    simple_benches = []
    int_data_path = "./data/INT-Duidelijke-Taal/CrowdsourcingResults.csv"
    amsterdam_simplification_path = (
        "./data/Amsterdam-Simplification/complex-simple-v1-anonymized.csv"
    )
    for prompt_type in ["detailed", "simple"]:
        simple_benches.append(
            INTDuidelijkeTaal(
                benchmark_name=f"INT_Duidelijke_Taal-{prompt_type}",
                data_path=int_data_path,
                prompt_type=prompt_type,
            )
        )

        simple_benches.append(
            AmsterdamSimplification(
                benchmark_name=f"AmsterdamSimplification-{prompt_type}",
                data_path=amsterdam_simplification_path,
                prompt_type=prompt_type,
            )
        )

    logging.info("Running comparison")
    leaderboard = Leaderboard(
        llms=[gpt, tinyllama],
        # llms=[tinyllama],
        benchmarks=[mmlu_nl_bench, arc_nl_bench] + simple_benches,
        codecarbon_params=codecarbon_params,
    )
    leaderboard.run_comparison(results_path="leaderboard")


if __name__ == "__main__":
    test_leaderboard()
