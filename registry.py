from unstructured_processor import DocumentProcessor
from structured_processor import StructuredFileProcessor


def build_registry(vectorstore, engine):
    documents = DocumentProcessor(vectorstore)
    structured = StructuredFileProcessor(engine)

    return {
        ".pdf":  documents,
        ".txt":  documents,
        ".md":   documents,
        ".docx": documents,
        ".csv":  structured,
        ".xlsx": structured,
        ".xls":  structured,
    }