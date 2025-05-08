"""Handle connection to key vault and retrieval of secrets"""
from azure.keyvault.secrets import SecretClient


class KeyVault:
    """Connect to the key vault using uri and DefaultAzureCredential"""

    def __init__(self, kv_uri, credential):
        self.keyvault_client = SecretClient(vault_url=kv_uri, credential=credential)

    def get_secret(self, secret_name):
        """Get a secret out of the vault given its name"""
        retrieved_secret = self.keyvault_client.get_secret(secret_name)
        return retrieved_secret.value
