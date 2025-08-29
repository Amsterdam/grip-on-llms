#!/usr/bin/env python3
"""
Comprehensive speed benchmark test for all models using vLLM with H100 profiles.
Tests each model with a standard prompt and measures inference speed.
"""
import json
import os
import time
from datetime import datetime

from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.language_models.model_router import LLMRouter
from tests.env_setup import get_hf_secrets


class VLLMSpeedBenchmark:
    """Comprehensive speed benchmark for all models with vLLM."""

    def __init__(self):
        self.test_prompt = (
            "What are the three most important factors for sustainable urban development?"
        )
        self.results = {}
        self.failed_models = {}

    def get_model_categories(self):
        """Categorize models by H100 profile."""
        categories = {"tiny": [], "small": [], "medium": [], "large": [], "xlarge": []}

        for model_name, config in MODEL_MAPPING.items():
            profile = config.get("h100_profile", "unknown")
            if profile in categories:
                categories[profile].append(model_name)

        return categories

    def test_single_model(self, model_name, timeout_seconds=120):  # noqa
        """Test a single model and measure performance."""
        print("\n{'='*60}")
        print(f"Testing: {model_name}")
        print("{'='*60}")
        hf_secrets = get_hf_secrets()

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
                hf_cache=os.environ["HF_CACHE"],
            )

            load_time = time.time() - load_start
            print(f"✓ Model loaded in {load_time:.2f}s")

            # Get model metadata
            metadata = model.get_metadata()
            print(f"Inference engine: {metadata.get('inference_engine', 'N/A')}")

            # Test inference speed
            print(f"Testing inference with prompt: '{self.test_prompt[:50]}...'")

            # Warmup run
            print("Performing warmup run...")
            warmup_start = time.time()
            try:
                _ = model.prompt(self.test_prompt)
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
                    response = model.prompt(self.test_prompt)
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
                estimated_input_tokens = len(self.test_prompt) / 3.5
                total_tokens = estimated_output_tokens + estimated_input_tokens
                tokens_per_second = total_tokens / avg_time if avg_time > 0 else 0
                output_tokens_per_second = (
                    estimated_output_tokens / avg_time if avg_time > 0 else 0
                )

                print("\n📊 Performance Summary:")
                print(f"  Average inference time: {avg_time:.2f}s")
                print(f"  Min/Max time: {min_time:.2f}s / {max_time:.2f}s")
                print(f"  Total tokens/sec: {tokens_per_second:.1f}")
                print(f"  Output tokens/sec: {output_tokens_per_second:.1f}")
                print(f"  Average response length: {avg_response_len:.0f} chars")
                print(f"  Estimated output tokens: {estimated_output_tokens:.0f}")

                # Store results
                self.results[model_name] = {
                    "model_id": model_id,
                    "h100_profile": h100_profile,
                    "load_time": load_time,
                    "warmup_time": warmup_time,
                    "avg_inference_time": avg_time,
                    "min_inference_time": min_time,
                    "max_inference_time": max_time,
                    "tokens_per_second": tokens_per_second,
                    "output_tokens_per_second": output_tokens_per_second,
                    "estimated_output_tokens": estimated_output_tokens,
                    "avg_response_length": avg_response_len,
                    "successful_runs": len(inference_times),
                    "sample_response": responses[0][:200] + "..." if responses else "",
                    "metadata": metadata,
                    "status": "success",
                }

                # Show sample response
                if responses:
                    print("\n📝 Sample response:")
                    print(f"  {responses[0][:150]}{'...' if len(responses[0]) > 150 else ''}")

            else:
                print("❌ All inference runs failed")
                self.failed_models[model_name] = "All inference runs failed"

            # Unload model to free memory
            print("\nUnloading model...")
            model.unload_model()

        except Exception as e:
            print(f"❌ Model test failed: {e}")
            self.failed_models[model_name] = str(e)
            model.unload_model()

    def run_all_models(self, profile_filter=None):
        """Run speed test on all models or filtered by profile."""
        categories = self.get_model_categories()

        print("🚀 vLLM Speed Benchmark - All Models with H100 Profiles")
        print("=" * 80)
        print(f"Test prompt: {self.test_prompt}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        if profile_filter:
            print(f"Filter: Testing only {profile_filter} profile models")
            test_models = categories.get(profile_filter, [])
        else:
            test_models = []
            for _, models in categories.items():
                test_models.extend(models)

        print(f"Total models to test: {len(test_models)}\n")

        # Test each model
        for i, model_name in enumerate(test_models, 1):
            print(f"\n[{i}/{len(test_models)}] Starting test for {model_name}")

            try:
                self.test_single_model(model_name)
            except KeyboardInterrupt:
                print("\n⚠ Test interrupted by user")
                break
            except Exception as e:
                print(f"❌ Unexpected error testing {model_name}: {e}")
                self.failed_models[model_name] = f"Unexpected error: {e}"

        # Generate summary
        self.generate_summary()

    def generate_summary(self):  # noqa
        """Generate comprehensive summary of results."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        successful_models = len(self.results)
        failed_models = len(self.failed_models)
        total_models = successful_models + failed_models

        print(f"Total models tested: {total_models}")
        print(f"Successful: {successful_models}")
        print(f"Failed: {failed_models}")

        if self.results:
            # Sort by performance
            sorted_results = sorted(
                self.results.items(),
                key=lambda x: x[1]["tokens_per_second"],
                reverse=True,
            )

            print("\n TOP PERFORMERS (by tokens/second):")
            print("-" * 60)
            for i, (model, data) in enumerate(sorted_results[:5], 1):
                print(
                    f"{i:2d}. {model:<25} {data['tokens_per_second']:6.1f} tok/s "
                    f"({data['avg_inference_time']:.2f}s avg)"
                )

            # Profile-based analysis
            print("\n PERFORMANCE BY H100 PROFILE:")
            print("-" * 60)

            profile_stats = {}
            for _, data in self.results.items():
                profile = data["h100_profile"]
                if profile not in profile_stats:
                    profile_stats[profile] = []
                profile_stats[profile].append(data["tokens_per_second"])

            for profile in ["tiny", "small", "medium", "large", "xlarge"]:
                if profile in profile_stats:
                    speeds = profile_stats[profile]
                    avg_speed = sum(speeds) / len(speeds)
                    print(
                        f"{profile.upper():<8}: {avg_speed:6.1f} tok/s avg "
                        f"({len(speeds)} models)"
                    )

            # Load time analysis
            print("\nLOAD TIMES:")
            print("-" * 60)

            load_times = sorted(
                [(model, data["load_time"]) for model, data in self.results.items()],
                key=lambda x: x[1],
            )

            print("Fastest loading:")
            for model, load_time in load_times[:3]:
                print(f"  {model:<25} {load_time:6.2f}s")

            if len(load_times) > 3:
                print("Slowest loading:")
                for model, load_time in load_times[-3:]:
                    print(f"  {model:<25} {load_time:6.2f}s")

        if self.failed_models:
            print("\n❌ FAILED MODELS:")
            print("-" * 60)
            for model, error in self.failed_models.items():
                print(f"{model}: {error[:80]}{'...' if len(error) > 80 else ''}")

        # Save detailed results
        self.save_results()

    def save_results(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"tests/vllm_speed_benchmark_{timestamp}.json"

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "test_prompt": self.test_prompt,
            "successful_models": len(self.results),
            "failed_models": len(self.failed_models),
            "results": self.results,
            "failures": self.failed_models,
        }

        try:
            with open(results_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {results_file}")
        except Exception as e:
            print(f"\nCould not save results file: {e}")


def main():
    """Run the speed benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Speed Benchmark for All Models")
    parser.add_argument(
        "--profile",
        choices=["tiny", "small", "medium", "large", "xlarge"],
        help="Test only models with specific H100 profile",
    )
    parser.add_argument("--model", help="Test only a specific model")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and their profiles",
    )

    args = parser.parse_args()

    benchmark = VLLMSpeedBenchmark()

    if args.list_models:
        print("Available models and their H100 profiles:")
        print("-" * 50)
        categories = benchmark.get_model_categories()
        for profile, models in categories.items():
            if models:
                print(f"\n{profile.upper()} Profile:")
                for model in sorted(models):
                    print(f"  - {model}")
        return

    if args.model:
        if args.model in MODEL_MAPPING:
            print(f"Testing single model: {args.model}")
            benchmark.test_single_model(args.model)
            benchmark.generate_summary()
        else:
            print(f"Model '{args.model}' not found in MODEL_MAPPING")
            return
    else:
        benchmark.run_all_models(profile_filter=args.profile)


if __name__ == "__main__":
    main()
