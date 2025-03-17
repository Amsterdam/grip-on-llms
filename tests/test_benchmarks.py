"""Basic tests to check whether a benchmark works as expected."""
import os

from llm_eval.benchmarks import ARC, MMLU
from llm_eval.language_models import LLMRouter


def test_mmlu_nl_gpt():
    """Test MMLU benchmark using GPT-4o model."""
    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        # "stop": None,
    }

    # Test GPT
    gpt = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        api_version=os.environ["API_VERSION"],
        params=gpt_params,
    )

    source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_mmlu/nl_dev.json"
    data_folder = "./data"
    mmlu_nl_bench = MMLU(source, data_folder, categories=["moral_disputes"])

    mmlu_nl_bench.eval(gpt, "gpt_results")


def test_mmlu_nl_hf():
    """Test MMLU benchmark using Tiny-llama model from Hugging Face."""
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
    tinyllama = LLMRouter.get_model(
        provider="huggingface",
        model_name="tiny-llama",
        hf_token=os.environ["HF_TOKEN"],
        #        hf_cache=os.environ["HF_CACHE"],
        hf_cache=None,
        params=hf_params,
    )

    source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_mmlu/nl_dev.json"
    data_folder = "./data"
    mmlu_nl_bench = MMLU(source, data_folder, categories=["moral_disputes"])

    mmlu_nl_bench.eval(tinyllama, "tinyllama_results")


def test_arc_nl_hf():
    """Test MMLU benchmark using Tiny llama model from Hugging Face."""
    hf_params = {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.65,
        "max_new_tokens": 200,
        "no_repeat_ngram_size": 3,
        "num_return_sequences": 1,
    }

    # Test HF Model
    tinyllama = LLMRouter.get_model(
        provider="huggingface",
        model_name="tiny-llama",
        hf_token=os.environ["HF_TOKEN"],
        hf_cache=None,
        params=hf_params,
    )

    source = "http://nlp.uoregon.edu/download/okapi-eval/datasets/m_arc/nl_validation.json"
    data_folder = "./data"
    arc_nl_bench = ARC(source, data_folder, categories=["LEAP"])

    arc_nl_bench.eval(tinyllama, "tinyllama_results")


if __name__ == "__main__":
    # test = "Test!"
    test_prompt = "Hoe maak ik een melding in Amsterdam?"

    # test_gpt(test_prompt)
    # test_hf(test_prompt)

    # test_mmlu_nl_gpt()
    # test_mmlu_nl_hf()

    test_arc_nl_hf()
