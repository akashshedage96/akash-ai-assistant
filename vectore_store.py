from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch

from config import AZURE_SEARCH_API_KEY, AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_INDEX_NAME, AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER
import logging


from unstructured_ingestion import create_chunks, create_chunk_ids
from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBEDDING_MODEL,
)
from storage import BlobStorageService


from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 32},
    )





def get_vectorstore():
    embeddings = get_embeddings()
    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_API_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings,
    )


logger = logging.getLogger(__name__)

BATCH_SIZE = 500   

def add_vectors(blob_name: str, content: bytes, etag: str) -> list[str]:
    
    vectorstore = get_vectorstore()

    chunks = create_chunks(blob_name, content, etag)
    if not chunks:
        logger.warning("No text found in %s", blob_name)
        return []

    keys = create_chunk_ids(blob_name, etag, len(chunks))
    stored_keys = []

    for i in range(0, len(chunks), BATCH_SIZE):
        stored_keys.extend(
            vectorstore.add_documents(
                chunks[i:i + BATCH_SIZE],
                keys=keys[i:i + BATCH_SIZE],
            )
        )

    logger.info("Added %d vectors for %s", len(stored_keys), blob_name)
    return stored_keys


DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
BATCH_SIZE = 500


def main():
    blob_service = BlobStorageService()
    blob_names = blob_service.collect_files()
    