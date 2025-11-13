"""Basic tests to test whether LLMs are working as expected"""
import logging

from env_setup import get_gpt_secrets, get_hf_secrets

from llm_eval.language_models import LLMRouter


def test_openai(test_prompt, model_name="gpt-4o"):
    logging.info("Testing GPT")

    gpt_secrets = get_gpt_secrets()

    gpt_params = {
        "frequency_penalty": 0,
        "presence_penalty": 0,
        # "stop": None,
    }

    # Test GPT
    model = LLMRouter.get_model(
        provider="azure",
        model_name=model_name,
        api_endpoint=gpt_secrets["API_ENDPOINT"],
        api_key=gpt_secrets["API_KEY"],
        api_version=gpt_secrets["API_VERSION"],
        params=gpt_params,
    )
    logging.info(f"GPT Response to {test_prompt}!: {model.prompt(test_prompt)}")
    logging.info(f"You can also call directly: {model(test_prompt)}")


def test_hf(test_prompt, model_name="tiny_llama"):
    logging.info(f"Testing HuggingFace: {model_name}")

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
    logging.info(f"HF Response to {test_prompt}!: {model.prompt(test_prompt)}")

    del model


if __name__ == "__main__":
    # test = "Test!"
    openai_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "o1-mini",
        "Llama-3-3-70B-Instruct",
        "Llama-4-Maverick-17B-128E-Instruct-FP8",
    ]

    test_prompt = "Hoe maak ik een melding in Amsterdam?"

    for model_name in openai_models:
        test_openai(test_prompt=test_prompt, model_name=model_name)
    # test_hf(test_prompt)

    models = [
        # "gpt-oss-120b",
        # "tiny-llama",
        # "phi-4-mini-instruct",
        # "llama-3.1-8b-instruct",
        # "falcon3-7b-instruct",
        # "mistral-7b-instruct-v0.3",
        # "llama-3.2-3b-instruct",
    ]

    for model_name in models:
        test_hf(test_prompt=test_prompt, model_name=model_name)
