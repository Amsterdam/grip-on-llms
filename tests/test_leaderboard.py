"""Basic tests to check whether a benchmark works as expected"""
import logging
from pathlib import Path

from env_setup import (
    base_results_folder,
    benchmark_data_folder,
    get_gpt_secrets,
    get_hf_secrets,
)

from llm_eval.benchmarks import (
    ARC,
    MMLU,
    AmsterdamSimplification,
    BZKSocialBias,
    CNNDailyMail,
    DutchBBQ,
    DutchCrowSPairs,
    HonestCityBench,
    INTDuidelijkeTaal,
    TinyARC,
    TinyMMLU,
    TinyTruthfulQA,
    XSum,
)
from llm_eval.language_models import LLMRouter
from llm_eval.leaderboard import Leaderboard
from llm_eval.translators import TranslatorRouter


def test_leaderboard():
    codecarbon_params = {
        "country_iso_code": "SWE",
        "region": "sweden",
        "allow_multiple_runs": True,
        "save_to_file": False,
        "pue": 1.185,
        # Directly set log level as suppressing logs doesn't work in other ways
        "log_level": "WARNING",
    }

    logging.info("Initializing GPT(s?)")
    gpt_secrets = get_gpt_secrets()

    gpt_translation_params = {
        # temp ~0-0.2 to avoid weird translations
        "temperature": 0.2,
        # top_p ~0.95-1
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
    }

    translation_gpt = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=gpt_secrets["API_ENDPOINT"],
        api_key=gpt_secrets["API_KEY"],
        api_version=gpt_secrets["API_VERSION"],
        params=gpt_translation_params,
        uses_api=True,
    )

    gpt_params = {
        "temperature": 0,
        "top_p": 1,
        # Don't penalize to ensure greedy decoding
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        # "stop": None,
        "max_tokens": 200,
    }

    gpt_4o = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o",
        api_endpoint=gpt_secrets["API_ENDPOINT"],
        api_key=gpt_secrets["API_KEY"],
        api_version=gpt_secrets["API_VERSION"],
        params=gpt_params,
        uses_api=True,
    )

    gpt_4o_mini = LLMRouter.get_model(
        provider="azure",
        model_name="gpt-4o-mini",
        api_endpoint=gpt_secrets["API_ENDPOINT"],
        api_key=gpt_secrets["API_KEY"],
        api_version=gpt_secrets["API_VERSION"],
        params=gpt_params,
        uses_api=True,
    )

    # gpt_35_turbo = LLMRouter.get_model(
    #     provider="azure",
    #     model_name="gpt-35-turbo",
    #     api_endpoint=gpt_secrets["API_ENDPOINT"],
    #     api_key=gpt_secrets["API_KEY"],
    #     api_version=gpt_secrets["API_VERSION"],
    #     params=gpt_params,
    #     uses_api=True,
    # )

    hf_secrets = get_hf_secrets()
    hf_inference_params = {
        "do_sample": False,
        # temp, top_k & top_p - unused for greedy decoding (adding for transparency)
        "temperature": 0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "num_return_sequences": 1,
        # "no_repeat_ngram_size": 3,
        "max_new_tokens": 200,
    }

    hf_object_params = {
        "provider": "vllm",
        # "provider": "huggingface",
        "hf_token": hf_secrets["HF_TOKEN"],
        "params": hf_inference_params,
        "uses_api": False,
    }

    logging.info("Initializing some HF model")
    tinyllama = LLMRouter.get_model(
        model_name="tiny-llama",
        **hf_object_params,
    )

    mistral = LLMRouter.get_model(
        # model_name="mistral-small",
        model_name="mistral-7b-instruct-v0.3",
        **hf_object_params,
    )

    mistral_small = LLMRouter.get_model(
        model_name="mistral-small-instruct",
        **hf_object_params,
    )

    mistral_large = LLMRouter.get_model(
        model_name="mistral-large-instruct",
        **hf_object_params,
    )

    mistral_large_quantized = LLMRouter.get_model(
        model_name="mistral-large-instruct-quantized",
        **hf_object_params,
    )

    llama = LLMRouter.get_model(
        model_name="llama-3.1-8b-instruct",
        **hf_object_params,
    )

    llama_large = LLMRouter.get_model(
        model_name="llama-3.3-70b-instruct",
        **hf_object_params,
    )
    llama_quantized = LLMRouter.get_model(model_name="Llama-3.3-70B-quantized", **hf_object_params)

    phi = LLMRouter.get_model(
        model_name="phi-4-mini-instruct",
        **hf_object_params,
    )

    falcon = LLMRouter.get_model(
        model_name="falcon3-7b-instruct",
        **hf_object_params,
    )

    olmo_small = LLMRouter.get_model(
        model_name="olmo-7b-instruct",
        **hf_object_params,
    )

    olmo_large = LLMRouter.get_model(
        model_name="olmo-32b-instruct",
        **hf_object_params,
    )

    eurollm_small = LLMRouter.get_model(
        model_name="eurollm-9b-instruct",
        **hf_object_params,
    )

    eurollm_large = LLMRouter.get_model(
        model_name="eurollm-22b-instruct",
        **hf_object_params,
    )

    qwen_small = LLMRouter.get_model(
        model_name="qwen-8b",
        **hf_object_params,
    )

    qwen_large = LLMRouter.get_model(
        model_name="qwen-32b",
        **hf_object_params,
    )

    gemma_small = LLMRouter.get_model(
        model_name="gemma-12b-instruct",
        **hf_object_params,
    )

    gemma_large = LLMRouter.get_model(
        model_name="gemma-27b-instruct",
        **hf_object_params,
    )

    logging.info("Setting up benchmarks")

    # Run mmlu using the local dump
    benchmark_name = "MMLU-NL"
    data_path = Path(benchmark_data_folder) / benchmark_name / "mmmlu_nl_dev.json"
    # mmlu_nl_bench = MMLU(benchmark_name, data_path=data_path, categories=["moral_disputes"])
    mmlu_nl_bench = MMLU(benchmark_name, data_path=data_path, categories=[])

    # Run arc using the local dump
    benchmark_name = "ARC-NL"
    data_path = Path(benchmark_data_folder) / benchmark_name / "marc_nl_validation.json"
    # arc_nl_bench = ARC(benchmark_name, data_path=data_path, categories=["LEAP"])
    arc_nl_bench = ARC(benchmark_name, data_path=data_path, categories=[])

    translation_model = translation_gpt
    # translation_model = tinyllama

    en_nl_translator = TranslatorRouter.get_translator(
        translator_type="llm_based",
        model_name=translation_model.model_name,
        llm=translation_model,
        source_lang="EN",
        target_lang="NL",
    )

    n_samples = 10

    simple_benches = []

    # for prompt_type in ["detailed", "simple"]:
    for prompt_type in ["detailed"]:
        int_data_path = (
            Path(benchmark_data_folder) / "INT-Duidelijke-Taal/CrowdsourcingResults.csv"
        )

        simple_benches.append(
            INTDuidelijkeTaal(
                benchmark_name=f"INT_Duidelijke_Taal-{prompt_type}",
                data_path=int_data_path,
                prompt_type=prompt_type,
            )
        )

        amsterdam_simplification_path = (
            Path(benchmark_data_folder)
            / "Amsterdam-Simplification/complex-simple-v1-anonymized.csv"
        )

        simple_benches.append(
            AmsterdamSimplification(
                benchmark_name=f"AmsterdamSimplification-{prompt_type}",
                data_path=amsterdam_simplification_path,
                prompt_type=prompt_type,
            )
        )

    summary_benches = []

    # for prompt_type in ["detailed", "simple"]:
    for prompt_type in ["detailed"]:
        # for language in ["NL", "EN"]
        for sum_lang in ["NL"]:
            bench_name = "CNNDailyMail"
            data_dir = Path(benchmark_data_folder) / bench_name
            summary_benches.append(
                CNNDailyMail(
                    benchmark_name=bench_name,
                    language=sum_lang,
                    prompt_type=prompt_type,
                    data_dir=data_dir,
                    translator=en_nl_translator,
                    max_translation_entries=n_samples,
                )
            )

            bench_name = "XSum"
            data_dir = Path(benchmark_data_folder) / bench_name
            summary_benches.append(
                XSum(
                    benchmark_name=bench_name,
                    language=sum_lang,
                    prompt_type=prompt_type,
                    data_dir=data_dir,
                    translator=en_nl_translator,
                    max_translation_entries=n_samples,
                )
            )

    tiny_benches = []

    tiny_benches_lang = "NL"

    tiny_benches.append(
        TinyMMLU(
            benchmark_name="TinyMMLU",
            language=tiny_benches_lang,
            data_dir=Path(benchmark_data_folder) / "TinyMMLU",
            translator=en_nl_translator,
        )
    )

    tiny_benches.append(
        TinyARC(
            benchmark_name="TinyARC",
            language=tiny_benches_lang,
            data_dir=Path(benchmark_data_folder) / "TinyARC",
            translator=en_nl_translator,
        )
    )

    tiny_benches.append(
        TinyTruthfulQA(
            benchmark_name="TinyTruthfulQA",
            language=tiny_benches_lang,
            data_dir=Path(benchmark_data_folder) / "TinyTruthfulQA",
            translator=en_nl_translator,
        )
    )

    bias_benches = []
    bias_benches.append(
        DutchBBQ(
            benchmark_name="Dutch-BBQ",
            language="NL",
            data_dir=Path(benchmark_data_folder) / "Dutch-BBQ",
        )
    )

    bias_benches.append(
        DutchCrowSPairs(
            benchmark_name="Dutch-CrowSPairs",
            language="NL",
            data_dir=Path(benchmark_data_folder) / "Dutch-CrowSPairs",
        )
    )

    bias_benches.append(
        BZKSocialBias(
            which_test="gender",
            benchmark_name="BZK-Social-Bias",
            language="NL",
            data_dir=Path(benchmark_data_folder) / "BZK-Social-Bias-gender",
        )
    )

    bias_benches.append(
        BZKSocialBias(
            which_test="name",
            benchmark_name="BZK-Social-Bias",
            language="NL",
            data_dir=Path(benchmark_data_folder) / "BZK-Social-Bias-name",
        )
    )

    # Run HonestCity; skip judges, eval later
    benchmark_name = "HonestCity"
    data_path = Path(benchmark_data_folder) / benchmark_name / "honest_city_final_annotated.xlsx"
    honest_city_bench = HonestCityBench(
        benchmark_name,
        data_path=data_path,
        llm_judges=[],
    )

    experiment_name = "vllm_rerun_2025-09-30"
    results_dir = Path(base_results_folder) / experiment_name

    logging.info("Running comparison")
    leaderboard = Leaderboard(
        llms=[
            mistral,
            mistral_small,
            mistral_large,
            mistral_large_quantized,
            llama,
            llama_large,
            llama_quantized,
            gpt_4o,
            gpt_4o_mini,
            falcon,
            phi,
            tinyllama,
            olmo_small,
            olmo_large,
            eurollm_small,
            eurollm_large,
            qwen_small,
            qwen_large,
            gemma_small,
            gemma_large,
        ],
        benchmarks=bias_benches
        + tiny_benches
        + simple_benches
        + summary_benches
        + [mmlu_nl_bench]
        + [arc_nl_bench]
        + [honest_city_bench],
        codecarbon_params=codecarbon_params,
        n_samples=n_samples,
    )
    leaderboard.run_comparison(results_dir=results_dir, results_path=None)


if __name__ == "__main__":
    test_leaderboard()
