"""
Comprehensive speed benchmark test for all models using vLLM with H100 profiles.
Tests each model with a standard prompt and measures inference speed.
"""
import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from tests.env_setup import get_hf_secrets


from llm_eval.language_models.llms.llm_config import MODEL_MAPPING
from llm_eval.language_models.model_router import LLMRouter


def get_test_results_file() -> Path:
    """Get the path to the test results JSON file in HF_HOME."""
    hf_secrets = get_hf_secrets()
    hf_cache = hf_secrets["HF_CACHE"]
    return Path(hf_cache) / "vllm_test_results.json"


def load_test_results() -> Dict:
    """Load existing test results from JSON file."""
    results_file = get_test_results_file()
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Could not load test results file: {e}")
            return {}
    return {}


def save_test_results(results: Dict):
    """Save test results to JSON file."""
    results_file = get_test_results_file()
    try:
        # Ensure directory exists
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
    except IOError as e:
        print(f"⚠️  Could not save test results: {e}")


def is_model_tested_successfully(model_name: str, results: Dict) -> bool:
    """Check if a model was tested successfully before."""
    return (
        model_name in results and 
        results[model_name].get("status") == "success" and
        results[model_name].get("avg_inference_time") is not None
    )


def test_single_model(model_name: str, save_results: bool = True) -> Dict:  # noqa
    """Test a single model and measure performance."""
    print(f"Testing: {model_name}")
    hf_secrets = get_hf_secrets()
    test_prompt = "Wat zijn de risico's van Large Language Models voor de overheid?"
    
    test_result = {
        "model_name": model_name,
        "status": "failed",
        "timestamp": datetime.now().isoformat(),
        "error": None,
        "traceback": None,
        "load_time": None,
        "avg_inference_time": None,
        "min_inference_time": None,
        "max_inference_time": None,
        "tokens_per_second": None,
        "output_tokens_per_second": None,
        "avg_response_length": None,
        "successful_runs": 0,
        "total_runs": 3,
    }
    
    model = None  # Initialize model variable for cleanup

    try:
        model_config = MODEL_MAPPING[model_name]
        model_id = model_config["id"]
        h100_profile = model_config.get("h100_profile", "auto")

        print(f"Model ID: {model_id}")
        print(f"H100 Profile: {h100_profile}")
        
        test_result["model_id"] = model_id
        test_result["h100_profile"] = h100_profile

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
        test_result["load_time"] = load_time

        # Get model metadata
        metadata = model.get_metadata()
        print(f"Inference engine: {metadata.get('inference_engine', 'N/A')}")
        test_result["inference_engine"] = metadata.get('inference_engine', 'N/A')

        # Test inference speed
        print(f"Testing inference with prompt: '{test_prompt[:50]}...'")

        # Warmup run
        print("Performing warmup run...")
        warmup_start = time.time()
        try:
            _ = model.prompt(test_prompt)
            warmup_time = time.time() - warmup_start
            print(f"✓ Warmup completed in {warmup_time:.2f}s")
            test_result["warmup_time"] = warmup_time
        except Exception as e:
            print(f"⚠ Warmup failed: {e}")
            test_result["warmup_error"] = str(e)

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

            # Update test result with metrics
            test_result.update({
                "status": "success",
                "avg_inference_time": avg_time,
                "min_inference_time": min_time,
                "max_inference_time": max_time,
                "tokens_per_second": tokens_per_second,
                "output_tokens_per_second": output_tokens_per_second,
                "avg_response_length": avg_response_len,
                "successful_runs": len(inference_times),
                "estimated_output_tokens": estimated_output_tokens,
                "sample_response": responses[0][:200] if responses else None
            })

            print("\n✅ Performance Summary:")
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
            test_result["error"] = "All inference runs failed"

    except Exception as e:
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"❌ Model test failed: {error_msg}")
        print(f"📍 Error details:")
        print(f"   Location: {traceback.extract_tb(e.__traceback__)[-1]}")
        print(f"   Full traceback saved to results")
        
        test_result.update({
            "error": error_msg,
            "traceback": error_traceback
        })
    
    finally:
        # Clean up model to free GPU memory
        try:
            if 'model' in locals() and model is not None:
                model.unload_model()
                print("🧹 Model unloaded from GPU")
        except Exception as cleanup_error:
            print(f"⚠️  Could not unload model: {cleanup_error}")
    
    # Save results if requested
    if save_results:
        try:
            all_results = load_test_results()
            all_results[model_name] = test_result
            save_test_results(all_results)
            print(f"💾 Test results saved for {model_name}")
        except Exception as save_error:
            print(f"⚠️  Could not save test results: {save_error}")
    
    return test_result


def test_all_models(skip_successful: bool = True) -> Dict:
    """Test all models from MODEL_MAPPING with optional skipping of successful tests."""
    print("🚀 Starting comprehensive vLLM testing for all models")
    print(f"📊 Found {len(MODEL_MAPPING)} models to test")
    
    # Load existing results for skip logic
    existing_results = load_test_results() if skip_successful else {}
    results_file = get_test_results_file()
    print(f"📁 Results will be stored in: {results_file}")
    
    if skip_successful and existing_results:
        successful_models = [
            name for name, result in existing_results.items() 
            if is_model_tested_successfully(name, existing_results)
        ]
        print(f"✅ Found {len(successful_models)} previously successful tests")
        if successful_models:
            print("   Previously successful models:")
            for model in successful_models:
                print(f"     • {model}")
    
    print("💡 Press Ctrl+C during a test to skip that model and continue")
    print("=" * 80)
    
    successful_tests: List[str] = []
    failed_tests: List[str] = []
    skipped_tests: List[str] = []
    
    for i, model_name in enumerate(MODEL_MAPPING.keys(), 1):
        print(f"\n🔄 [{i}/{len(MODEL_MAPPING)}] Processing: {model_name}")
        
        # Check if we should skip this model
        if skip_successful and is_model_tested_successfully(model_name, existing_results):
            print(f"   ⏭️  Skipping: {model_name} (previously successful)")
            skipped_tests.append(model_name)
            continue
        
        try:
            result = test_single_model(model_name, save_results=True)
            if result["status"] == "success":
                successful_tests.append(model_name)
                print(f"   ✅ {model_name} - SUCCESS")
            else:
                failed_tests.append(model_name)
                print(f"   ❌ {model_name} - FAILED: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print(f"   ⏭️  Skipping: {model_name} (user interrupted)")
            skipped_tests.append(model_name)
            continue
            
        except Exception as e:
            print(f"   💥 {model_name} - CRASHED: {e}")
            failed_tests.append(model_name)
            # Log the crash
            crash_result = {
                "model_name": model_name,
                "status": "crashed", 
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            try:
                all_results = load_test_results()
                all_results[model_name] = crash_result
                save_test_results(all_results)
            except Exception:
                pass  # Don't fail the whole test run if we can't save
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    total_models = len(MODEL_MAPPING)
    print(f"📊 Total models: {total_models}")
    print(f"✅ Successful tests: {len(successful_tests)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    print(f"⏭️  Skipped tests: {len(skipped_tests)}")
    
    if successful_tests:
        print(f"\n✅ Successful tests ({len(successful_tests)}):")
        for model in successful_tests:
            print(f"   • {model}")
    
    if failed_tests:
        print(f"\n❌ Failed tests ({len(failed_tests)}):")
        for model in failed_tests:
            print(f"   • {model}")
    
    if skipped_tests:
        print(f"\n⏭️  Skipped tests ({len(skipped_tests)}):")
        for model in skipped_tests:
            print(f"   • {model}")
    
    print(f"\n📁 Full test results stored in: {results_file}")
    
    # Load and return final results
    return load_test_results()


def main():
    """Run the speed benchmark."""
    parser = argparse.ArgumentParser(description="vLLM Speed Benchmark for All Models")
    parser.add_argument("--model", help="Test only a specific model")
    parser.add_argument("--all", action="store_true", help="Test all models from MODEL_MAPPING")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip previously successful tests")
    args = parser.parse_args()

    if args.all:
        print("🚀 Testing all models from MODEL_MAPPING")
        skip_successful = not args.no_skip
        results = test_all_models(skip_successful=skip_successful)
        
        # Show final summary stats
        successful = sum(1 for r in results.values() if r.get("status") == "success")
        failed = sum(1 for r in results.values() if r.get("status") in ["failed", "crashed"])
        print(f"\n🏁 Final Results: {successful} successful, {failed} failed out of {len(MODEL_MAPPING)} total models")
        
    elif args.model:
        if args.model in MODEL_MAPPING:
            print(f"Testing single model: {args.model}")
            result = test_single_model(args.model)
            if result["status"] == "success":
                print(f"\n✅ {args.model} test completed successfully!")
            else:
                print(f"\n❌ {args.model} test failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"Model '{args.model}' not found in MODEL_MAPPING")
            print("Available models:")
            for model_name in MODEL_MAPPING.keys():
                print(f"  • {model_name}")
            return
    else:
        print("Please specify either --model <model_name> or --all")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
