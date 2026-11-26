from processor import BaseProcessor
from unstructured_ingestion import create_chunks, create_chunk_ids

BATCH_SIZE = 50   


class DocumentProcessor(BaseProcessor):

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def process(self, blob_name, content, etag):
        chunks = create_chunks(blob_name, content, etag)
        if not chunks:
            return []

        wanted_ids = create_chunk_ids(blob_name, etag, len(chunks))
        stored_ids = []

        for i in range(0, len(chunks), BATCH_SIZE):
            returned = self.vectorstore.add_documents(
                chunks[i:i + BATCH_SIZE],
                keys=wanted_ids[i:i + BATCH_SIZE],
            )
            stored_ids.extend(returned)       # the real keys used in the index

        return stored_ids

    def delete(self, blob_name, stored_ids):
        for i in range(0, len(stored_ids), BATCH_SIZE):
            self.vectorstore.delete(ids=stored_ids[i:i + BATCH_SIZE])