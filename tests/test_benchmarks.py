"""Basic tests to check whether a benchmark works as expected."""
import logging

from env_setup import get_gpt_secrets, get_hf_secrets

from llm_eval.benchmarks import ARC, MMLU, AmsterdamSimplification, INTDuidelijkeTaal
from llm_eval.language_models import LLMRouter


def get_gpt():
    """Get a gpt instance"""
    gpt_secrets = get_gpt_secrets()

    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        # "stop": None,
    }

    # Test GPT
    gpt = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=gpt_secrets["API_ENDPOINT"],
        api_key=gpt_secrets["API_KEY"],
        api_version=gpt_secrets["API_VERSION"],
        params=gpt_params,
    )

    return gpt


def get_hf_model(model_name="tiny-llama"):
    """Get huggingface model instance"""
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

    # Test HF Model
    model = LLMRouter.get_model(
        provider="huggingface",
        model_name=model_name,
        hf_token=hf_secrets["HF_TOKEN"],
        #        hf_cache=os.environ["HF_CACHE"],
        hf_cache=None,
        params=hf_params,
    )

    return model


def test_mmlu_nl_gpt():
    """Test MMLU benchmark using GPT-4o model."""
    logging.info("Testing MMLU using GPT")

    gpt = get_gpt()

    source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_mmlu/nl_dev.json"
    data_folder = "./data"
    mmlu_nl_bench = MMLU("MMLU-NL", source, data_folder, categories=["moral_disputes"])

    mmlu_nl_bench.eval(gpt, "results_gpt_mmlu")


def test_mmlu_nl_hf():
    """Test MMLU benchmark using Tiny-llama model from Hugging Face."""
    logging.info("Testing MMLU using a HuggingFace model")

    tinyllama = get_hf_model("tiny-llama")

    source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_mmlu/nl_dev.json"
    data_folder = "./data"
    mmlu_nl_bench = MMLU("MMLU-NL", source, data_folder, categories=["moral_disputes"])

    mmlu_nl_bench.eval(tinyllama, "results_tinyllama_mmlu")


def test_arc_nl_hf():
    """Test MMLU benchmark using Tiny llama model from Hugging Face."""
    logging.info("Testing ARC using a HuggingFace model")

    tinyllama = get_hf_model("tiny-llama")

    source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_arc/nl_validation.json"
    data_folder = "./data"
    arc_nl_bench = ARC("ARC-NL", source, data_folder, categories=["LEAP"])

    arc_nl_bench.eval(tinyllama, "results_tinyllama_arc")


def test_simplification_gpt():
    """Test simplification benchmarks using GPT-4o model."""
    logging.info("Testing MMLU using GPT")

    gpt = get_gpt()

    prompt_type = "detailed"
    int_data_path = "./data/INT-Duidelijke-Taal/CrowdsourcingResults.csv"
    int_simplification_bench = INTDuidelijkeTaal(
        benchmark_name=f"INT_Duidelijke_Taal-{prompt_type}",
        data_path=int_data_path,
        prompt_type=prompt_type,
    )

    amsterdam_simplification_path = (
        "./data/Amsterdam-Simplification/complex-simple-v1-anonymized.csv"
    )
    amsterdam_simplification_bench = AmsterdamSimplification(
        benchmark_name=f"AmsterdamSimplification-{prompt_type}",
        data_path=amsterdam_simplification_path,
        prompt_type=prompt_type,
    )

    int_simplification_bench.eval(gpt, "results_gpt_simple-int")
    amsterdam_simplification_bench.eval(gpt, "results_gpt_simple-city")


if __name__ == "__main__":
    test_mmlu_nl_gpt()
    test_mmlu_nl_hf()
    test_arc_nl_hf()
    test_simplification_gpt()
