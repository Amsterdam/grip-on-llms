"""Basic tests to test whether LLMs are working as expected"""
import os

from llm_eval.language_models import LLMRouter


def test_gpt(test_prompt):
    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        # "stop": None,
    }

    # Test GPT
    model = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        api_version=os.environ["API_VERSION"],
        params=gpt_params,
    )
    print(f"GPT Response to {test_prompt}!: {model.prompt(test_prompt)}")
    print(f"Second attempt {model(test_prompt)}")


def test_hf(test_prompt):
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
        model_name="tiny-llama",
        hf_token=os.environ["HF_TOKEN"],
        #        hf_cache=os.environ["HF_CACHE"],
        hf_cache=None,
        params=hf_params,
    )
    print(f"HF Response to {test_prompt}!: {model.prompt(test_prompt)}")


if __name__ == "__main__":
    # test = "Test!"
    test_prompt = "Hoe maak ik een melding in Amsterdam?"

    test_gpt(test_prompt)
    test_hf(test_prompt)
