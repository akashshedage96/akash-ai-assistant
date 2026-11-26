from processor import BaseProcessor
from structured_ingestion import StructuredFileIngestor


class StructuredFileProcessor(BaseProcessor):
    "CSV / Excel processor used by the pipeline"

    def __init__(self, engine):
        self.ingestion = StructuredFileIngestor(engine)

    def process(self, blob_name: str, content: bytes, etag: str) -> list[str]:
        table_name = self.ingestion.ingest_structured_file(blob_name, content)
        return [table_name]

    def delete(self, blob_name: str, stored_ids: list[str]) -> None:
        self.ingestion.delete_structured_file(blob_name)