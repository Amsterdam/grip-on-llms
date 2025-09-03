from tests.env_setup import get_hf_secrets
from huggingface_hub import snapshot_download
import argparse
import sys


def download_model(url):
    """Download model without loading it from huggingface"""
    try:
        print(f"Downloading model: {url}")
        snapshot_download(url)
        print(f"Successfully downloaded: {url}")
    except Exception as e:
        print(f"Error downloading model {url}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    get_hf_secrets()


    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub")
    parser.add_argument("url", type=str, help="Name of model to download, eg 'openai/gpt-oss-20b'")

    args = parser.parse_args()
    download_model(args.url)