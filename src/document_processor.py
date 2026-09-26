"""
Document ingestion layer.

Responsible for validating uploads, loading them into LangChain `Document`
objects regardless of source format, and splitting them into retrieval-ready
chunks. Kept isolated from the vector store and the LLM so each concern can
be tested and swapped independently.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings
from src.exceptions import EmptyDocumentError, FileTooLargeError, UnsupportedFileTypeError
from src.logging_config import get_logger

logger = get_logger(__name__)

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
}


@dataclass(frozen=True)
class UploadedFile:
    """A thin, framework-agnostic wrapper around an uploaded file's bytes."""

    name: str
    data: bytes


class DocumentProcessor:
    """Validates, loads, and chunks documents for indexing."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _validate(self, file: UploadedFile) -> str:
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in self._settings.allowed_extensions:
            raise UnsupportedFileTypeError(
                f"'{ext}' is not supported. Allowed types: "
                f"{', '.join(self._settings.allowed_extensions)}"
            )

        size_mb = len(file.data) / (1024 * 1024)
        if size_mb > self._settings.max_upload_mb:
            raise FileTooLargeError(
                f"'{file.name}' is {size_mb:.1f} MB, which exceeds the "
                f"{self._settings.max_upload_mb} MB limit."
            )
        return ext

    def load(self, file: UploadedFile) -> list[Document]:
        """Load a single uploaded file into LangChain Documents."""
        ext = self._validate(file)
        loader_cls = _LOADER_MAP[ext]

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.data)
            tmp_path = tmp.name

        try:
            loader = loader_cls(tmp_path)
            documents = loader.load()
        finally:
            os.unlink(tmp_path)

        if not documents or not any(doc.page_content.strip() for doc in documents):
            raise EmptyDocumentError(f"No extractable text found in '{file.name}'.")

        for doc in documents:
            doc.metadata["source"] = file.name

        logger.info("Loaded %d page(s)/section(s) from %s", len(documents), file.name)
        return documents

    def load_many(self, files: list[UploadedFile]) -> list[Document]:
        """Load and concatenate multiple uploaded files, skipping bad ones with a warning."""
        all_docs: list[Document] = []
        for file in files:
            try:
                all_docs.extend(self.load(file))
            except (UnsupportedFileTypeError, FileTooLargeError, EmptyDocumentError) as exc:
                logger.warning("Skipping '%s': %s", file.name, exc)
        return all_docs

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into overlapping, retrieval-sized chunks."""
        chunks = self._splitter.split_documents(documents)
        logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
        return chunks

    def process(self, files: list[UploadedFile]) -> list[Document]:
        """Full pipeline: validate -> load -> split."""
        documents = self.load_many(files)
        if not documents:
            raise EmptyDocumentError("None of the uploaded files produced usable text.")
        return self.split(documents)

