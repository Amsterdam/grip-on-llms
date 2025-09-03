from tests.env_setup import get_hf_secrets
from huggingface_hub import snapshot_download
import argparse


def download_model(url):
    """ Download model without loading it from huggingface"""
    snapshot_download(url)


if __name__ == "__main__":
    get_hf_secrets()
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str, help="Name of model to download, eg openai/gpt-oss-20b")
    args = parser.parse_args()
    download_model(args.url)
