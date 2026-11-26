import logging
import os

from azure.storage.blob import BlobServiceClient

from config import AZURE_STORAGE_CONNECTION_STRING, BLOB_CONTAINER

logger = logging.getLogger(__name__)


class BlobStorageService:

    def __init__(self, supported_extensions):
        self.supported_extensions = set(supported_extensions)
        self.blob_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_client = self.blob_client.get_container_client(BLOB_CONTAINER)

    @staticmethod
    def extract_file_extension(filename: str) -> str:
        return os.path.splitext(filename)[1].lower()

    def collect_files(self):
        "Yields name, etag, size, extension for each supported blob"
        try:
            for blob in self.container_client.list_blobs():
                ext = self.extract_file_extension(blob.name)
                if ext in self.supported_extensions:
                    yield {
                        "name": blob.name,
                        "etag": (blob.etag or "").strip('"'),
                        "size": blob.size,
                        "extension": ext,
                    }
        except Exception:
            logger.exception("Error listing blobs")
            raise

    def download_file(self, blob_name: str) -> bytes:
        return self.container_client.get_blob_client(blob_name).download_blob().readall()