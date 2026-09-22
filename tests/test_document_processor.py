import pytest

from src.config import Settings
from src.document_processor import DocumentProcessor, UploadedFile
from src.exceptions import EmptyDocumentError, FileTooLargeError, UnsupportedFileTypeError


@pytest.fixture
def settings() -> Settings:
    return Settings(groq_api_key="test-key", max_upload_mb=1)


@pytest.fixture
def processor(settings: Settings) -> DocumentProcessor:
    return DocumentProcessor(settings)


def test_rejects_unsupported_extension(processor: DocumentProcessor):
    bad_file = UploadedFile(name="malware.exe", data=b"binary-data")
    with pytest.raises(UnsupportedFileTypeError):
        processor.load(bad_file)


def test_rejects_oversized_file(processor: DocumentProcessor):
    too_big = UploadedFile(name="big.txt", data=b"x" * (2 * 1024 * 1024))
    with pytest.raises(FileTooLargeError):
        processor.load(too_big)


def test_loads_plain_text_file(processor: DocumentProcessor):
    file = UploadedFile(name="notes.txt", data=b"This is a test document about RAG pipelines.")
    docs = processor.load(file)
    assert len(docs) == 1
    assert "RAG pipelines" in docs[0].page_content
    assert docs[0].metadata["source"] == "notes.txt"


def test_rejects_empty_document(processor: DocumentProcessor):
    file = UploadedFile(name="empty.txt", data=b"   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        processor.load(file)


def test_split_produces_chunks(processor: DocumentProcessor):
    file = UploadedFile(name="doc.txt", data=(b"Sentence about topic A. " * 200))
    docs = processor.load(file)
    chunks = processor.split(docs)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= processor._settings.chunk_size + 200 for c in chunks)


def test_load_many_skips_bad_files_gracefully(processor: DocumentProcessor):
    good = UploadedFile(name="good.txt", data=b"Valid content here.")
    bad = UploadedFile(name="bad.exe", data=b"nope")
    docs = processor.load_many([good, bad])
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "good.txt"

