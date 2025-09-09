"""
Comprehensive speed benchmark test for all models using vLLM with H100 profiles.
Tests each model with a standard prompt and measures inference speed.
"""
import argparse
import time

from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.language_models.model_router import LLMRouter
from tests.env_setup import get_hf_secrets


def test_single_model(model_name):  # noqa
    """Test a single model and measure performance."""
    print(f"Testing: {model_name}")
    hf_secrets = get_hf_secrets()
    test_prompt = "What are the three most important factors for sustainable urban development?"

    try:
        model_config = MODEL_MAPPING[model_name]
        model_id = model_config["id"]
        h100_profile = model_config.get("h100_profile", "auto")

        print(f"Model ID: {model_id}")
        print(f"H100 Profile: {h100_profile}")

        # Create model with vLLM and H100 optimization
        print("Loading model with vLLM...")
        load_start = time.time()

        model = LLMRouter.get_model(
            model_name=model_name,
            provider="vllm",
            hf_token=hf_secrets["HF_TOKEN"],
        )

        load_time = time.time() - load_start
        print(f"✓ Model loaded in {load_time:.2f}s")

        # Get model metadata
        metadata = model.get_metadata()
        print(f"Inference engine: {metadata.get('inference_engine', 'N/A')}")

        # Test inference speed
        print(f"Testing inference with prompt: '{test_prompt[:50]}...'")

        # Warmup run
        print("Performing warmup run...")
        warmup_start = time.time()
        try:
            _ = model.prompt(test_prompt)
            warmup_time = time.time() - warmup_start
            print(f"✓ Warmup completed in {warmup_time:.2f}s")
        except Exception as e:
            print(f"⚠ Warmup failed: {e}")
            warmup_time = None

        # Actual benchmark runs
        inference_times = []
        responses = []

        for i in range(3):  # 3 runs for average
            print(f"Run {i+1}/3...")
            start_time = time.time()

            try:
                response = model.prompt(test_prompt)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                responses.append(response)

                print(f"  ✓ Completed in {inference_time:.2f}s")
                print(f"  Response length: {len(response)} chars")

            except Exception as e:
                print(f"  ✗ Run {i+1} failed: {e}")
                continue

        # Calculate statistics
        if inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            min_time = min(inference_times)
            max_time = max(inference_times)

            # Estimate tokens per second (rough estimate: 1 token ≈ 3.5 characters for English)
            avg_response_len = sum(len(r) for r in responses) / len(responses)
            estimated_output_tokens = avg_response_len / 3.5
            # Also estimate input tokens (prompt tokens)
            estimated_input_tokens = len(test_prompt) / 3.5
            total_tokens = estimated_output_tokens + estimated_input_tokens
            tokens_per_second = total_tokens / avg_time if avg_time > 0 else 0
            output_tokens_per_second = estimated_output_tokens / avg_time if avg_time > 0 else 0

            print("\n Performance Summary:")
            print(f"  Average inference time: {avg_time:.2f}s")
            print(f"  Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
            print(f"  Total tokens/sec: {tokens_per_second:.1f}")
            print(f"  Output tokens/sec: {output_tokens_per_second:.1f}")
            print(f"  Average response length: {avg_response_len:.0f} chars")
            print(f"  Estimated output tokens: {estimated_output_tokens:.0f}")

            # Show sample response
            if responses:
                print("\n Sample response:")
                print(f"  {responses[0][:150]}{'...' if len(responses[0]) > 150 else ''}")

        else:
            print("❌ All inference runs failed")

    except Exception as e:
        print(f"❌ Model test failed: {e}")


def main():
    """Run the speed benchmark."""
    parser = argparse.ArgumentParser(description="vLLM Speed Benchmark for All Models")
    parser.add_argument("--model", required=True, help="Test only a specific model")
    args = parser.parse_args()

    if args.model in MODEL_MAPPING:
        print(f"Testing single model: {args.model}")
        test_single_model(args.model)
    else:
        print(f"Model '{args.model}' not found in MODEL_MAPPING")
        return


if __name__ == "__main__":
    main()
