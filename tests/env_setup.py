"""Setup key vault connection and azure authentication"""
import json
import logging
import os
import subprocess

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from llm_eval.utils import KeyVault

logging.basicConfig(level=logging.INFO)

# Suppress logs from azure
azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.WARNING)

# Suppress logs from requests
requests_logger = logging.getLogger("httpx")
requests_logger.setLevel(logging.WARNING)


logging.info("Loading environment")
load_dotenv()

logging.info("Creating an InteractiveBrowserCredential instance")
azure_credential = DefaultAzureCredential()

logging.info("Setting up the key vault connection")
kv_uri = os.getenv("KVUri")
key_vault = KeyVault(kv_uri, azure_credential)

# Change HuggingFace cache to shared storage account folder
hf_cache = key_vault.get_secret("gp-hf-cache")
os.environ["HF_HOME"] = hf_cache

os.environ["TOKENIZERS_PARALLELISM"] = "false"

benchmark_data_folder = key_vault.get_secret("gp-shared-benchmark-data-path")


def get_hf_secrets():
    logging.info("Getting HuggingFace secrets")

    # Run the Azure CLI command to get account information & parse output
    azure_account = subprocess.run(["az", "account", "show"], stdout=subprocess.PIPE)
    azure_account_info = json.loads(azure_account.stdout)

    # Extract the username and turn into keyvault extention
    username = azure_account_info.get("user", {}).get("name")
    key_vault_name = username.split("@")[0].replace(".", "")

    # Get key vault extension
    hf_token_key = f"hf-token-{key_vault_name}"
    hf_token = key_vault.get_secret(hf_token_key)

    return {"HF_TOKEN": hf_token}


def get_gpt_secrets():
    logging.info("Getting GPT secrets")

    # Define the Azure OpenAI scope and obtain access token for it
    cs_scope = "https://cognitiveservices.azure.com/.default"
    cs_token = azure_credential.get_token(cs_scope).token

    api_endpoint = key_vault.get_secret("gp-openai-endpoint")
    api_version = key_vault.get_secret("gp-openai-api-version")

    return {
        "API_ENDPOINT": api_endpoint,
        "API_KEY": cs_token,
        "API_VERSION": api_version,
    }
