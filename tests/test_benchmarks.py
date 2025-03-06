"""Basic tests to check whether a benchmark works as expected"""
import os

from llm_eval.benchmarks import MMLU
from llm_eval.language_models import LLMRouter


def test_mmlu_nl():
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


if __name__ == "__main__":
    # test = "Test!"
    test_prompt = "Hoe maak ik een melding in Amsterdam?"

    # test_gpt(test_prompt)
    # test_hf(test_prompt)

    test_mmlu_nl()
