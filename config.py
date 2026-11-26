
import os
from dotenv import load_dotenv

load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"Missing required environment variable: {name}"
        )

    return value


AZURE_STORAGE_CONNECTION_STRING = get_required_env(
    "AZURE_STORAGE_CONNECTION_STRING"
)

BLOB_CONTAINER = get_required_env("BLOB_CONTAINER")

AZURE_SEARCH_ENDPOINT = get_required_env(
    "AZURE_SEARCH_ENDPOINT"
)

AZURE_SEARCH_API_KEY = get_required_env(
    "AZURE_SEARCH_API_KEY"
)

AZURE_SEARCH_INDEX_NAME = get_required_env(
    "AZURE_SEARCH_INDEX_NAME"
)

OPENROUTER_API_KEY = get_required_env(
    "OPENROUTER_API_KEY"
)

OPENROUTER_EMBEDDING_MODEL = get_required_env(
    "OPENROUTER_EMBEDDING_MODEL"
)

OPENROUTER_LLM_MODEL = get_required_env(
    "OPENROUTER_LLM_MODEL"
)

OPENROUTER_BASE_URL = get_required_env(
    "OPENROUTER_BASE_URL"
)

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200

SQL_DRIVER = "ODBC Driver 17 for SQL Server"
SQL_SERVER = r"(localdb)\MSSQLLocalDB"
SQL_DATABASE = "IngestionState"