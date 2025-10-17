"""Download a model from huggingface without loading it, possible on a cpu"""
import argparse
import logging
import os
import sys
from typing import Dict, List

from huggingface_hub import snapshot_download

from tests.env_setup import get_hf_secrets
from llm_eval.language_models.llms.llm_config import MODEL_MAPPING


def download_model(url):
    """Download model without loading it from huggingface"""
    try:
        logging.info(f"Downloading model: {url}")
        snapshot_download(url)
        logging.info(f"Successfully downloaded: {url}")
    except Exception as e:
        logging.error(f"Error downloading model {url}: {e}")
        sys.exit(1)


def get_cache_directory():
    """Get the HuggingFace cache directory"""
    # Check for custom HF_HOME first, then default cache
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        return cache_dir
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def download_all_models_from_config():
    """Download all models from llm_config.py MODEL_MAPPING"""
    successful_downloads: List[str] = []
    failed_downloads: List[Dict[str, str]] = []
    skipped_downloads: List[str] = []
    
    cache_dir = get_cache_directory()
    print("🤖 Starting download of all models from llm_config.py")
    print(f"📊 Found {len(MODEL_MAPPING)} models to download")
    print(f"📁 Models will be stored in: {cache_dir}")
    print("💡 Press Ctrl+C during a download to skip that model and continue")
    print("=" * 70)
    
    for model_name, model_config in MODEL_MAPPING.items():
        hub_id = model_config["id"]
        model_size = model_config["model_size"]
        
        print(f"\n📥 Downloading: {model_name}")
        print(f"   Hub ID: {hub_id}")
        print(f"   Size category: {model_size}")
        
        try:
            snapshot_download(hub_id)
            successful_downloads.append(model_name)
            print(f"   ✅ Successfully downloaded: {model_name}")
            
        except KeyboardInterrupt:
            skipped_downloads.append(model_name)
            print(f"   ⏭️  Skipped: {model_name} (user interrupted)")
            continue
            
        except Exception as e:
            error_info = {"name": model_name, "hub_id": hub_id, "error": str(e)}
            failed_downloads.append(error_info)
            print(f"   ❌ Failed to download {model_name}: {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 70)
    
    total_models = len(MODEL_MAPPING)
    print(f"📊 Total models: {total_models}")
    print(f"✅ Successfully downloaded: {len(successful_downloads)}")
    print(f"⏭️  Skipped: {len(skipped_downloads)}")
    print(f"❌ Failed: {len(failed_downloads)}")
    
    if successful_downloads:
        print(f"\n✅ Successful downloads ({len(successful_downloads)}):")
        for model in successful_downloads:
            print(f"   • {model}")
    
    if skipped_downloads:
        print(f"\n⏭️  Skipped downloads ({len(skipped_downloads)}):")
        for model in skipped_downloads:
            print(f"   • {model}")
    
    if failed_downloads:
        print(f"\n❌ Failed downloads ({len(failed_downloads)}):")
        for failure in failed_downloads:
            print(f"   • {failure['name']} ({failure['hub_id']})")
            print(f"     Error: {failure['error']}")
    
    print(f"\n📁 All downloaded models are stored in: {cache_dir}")
    
    if failed_downloads:
        print(f"\n⚠️  {len(failed_downloads)} models failed to download. Check errors above.")
        return False
    else:
        completion_msg = "🎉 All models downloaded successfully!" if not skipped_downloads else f"✅ Download completed! ({len(skipped_downloads)} models were skipped)"
        print(f"\n{completion_msg}")
        return True


if __name__ == "__main__":
    get_hf_secrets()

    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub")
    parser.add_argument("url", type=str, nargs='?', help="Name of model to download, eg 'openai/gpt-oss-20b'")
    parser.add_argument("--all", action="store_true", help="Download all models from llm_config.py")

    args = parser.parse_args()
    
    if args.all:
        success = download_all_models_from_config()
        sys.exit(0 if success else 1)
    elif args.url:
        download_model(args.url)
    else:
        print("Please provide either a model URL or use --all flag to download all models")
        parser.print_help()
        sys.exit(1)
