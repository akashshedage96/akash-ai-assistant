
import hashlib
import io

import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_document(blob_name: str, content: bytes):
    name = blob_name.lower()

    if name.endswith(".pdf"):
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".pdf"
        ) as temp:
            temp.write(content)
            temp.flush()
            return PyPDFLoader(temp.name).load()

    if name.endswith(".docx"):
        import tempfile

        with tempfile.NamedTemporaryFile(
            suffix=".docx"
        ) as temp:
            temp.write(content)
            temp.flush()
            return Docx2txtLoader(temp.name).load()

    if name.endswith((".txt", ".md")):
        text = content.decode("utf-8-sig")
        return [Document(page_content=text)]

    raise ValueError(
        f"Unsupported document format: {blob_name}"
    )
import hashlib
import os
import tempfile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


def _load_with_tempfile(loader_cls, suffix, content):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = temp.name
        return loader_cls(temp_path).load()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def load_document(blob_name, content):
    name = blob_name.lower()
    if name.endswith(".pdf"):
        return _load_with_tempfile(PyPDFLoader, ".pdf", content)
    if name.endswith(".docx"):
        return _load_with_tempfile(Docx2txtLoader, ".docx", content)
    if name.endswith((".txt", ".md")):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("cp1252", errors="replace")
        return [Document(page_content=text)]
    raise ValueError(f"Unsupported document format: {blob_name}")


def create_chunks(blob_name, content, etag):
    chunks = _splitter.split_documents(load_document(blob_name, content))
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = blob_name
        chunk.metadata["source_etag"] = etag
        chunk.metadata["chunk_number"] = i
    return chunks


def create_chunk_ids(blob_name, etag, count):
    return [
        hashlib.sha256(f"{blob_name}|{etag}|{i}".encode("utf-8")).hexdigest()
        for i in range(count)
    ]

def create_chunks(blob_name: str, content: bytes, etag: str):
    pages = load_document(blob_name, content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = blob_name
        chunk.metadata["source_etag"] = etag
        chunk.metadata["chunk_number"] = i

    return chunks


def create_chunk_ids(blob_name: str, etag: str, count: int):
    ids = []

    for i in range(count):
        raw = f"{blob_name}|{etag}|{i}"
        ids.append(
            hashlib.sha256(raw.encode("utf-8")).hexdigest()
        )

    return ids
