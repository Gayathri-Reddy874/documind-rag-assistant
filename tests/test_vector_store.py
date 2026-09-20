from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import Settings
from src.exceptions import VectorStoreNotReadyError
from src.vector_store import VectorStoreManager


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(groq_api_key="test-key", vector_store_dir=str(tmp_path))


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(page_content="LangChain simplifies building LLM applications.", metadata={"source": "a.txt"}),
        Document(page_content="FAISS is a fast vector similarity search library.", metadata={"source": "b.txt"}),
    ]


@patch("src.vector_store._load_embeddings")
def test_raises_before_build(mock_embeddings, settings):
    mock_embeddings.return_value = MagicMock()
    manager = VectorStoreManager(settings)
    assert manager.is_ready is False
    with pytest.raises(VectorStoreNotReadyError):
        manager.as_retriever()


@patch("src.vector_store.FAISS")
@patch("src.vector_store._load_embeddings")
def test_build_sets_ready(mock_embeddings, mock_faiss, settings, sample_docs):
    mock_embeddings.return_value = MagicMock()
    mock_faiss.from_documents.return_value = MagicMock()

    manager = VectorStoreManager(settings)
    manager.build(sample_docs)

    assert manager.is_ready is True
    mock_faiss.from_documents.assert_called_once()


@patch("src.vector_store.FAISS")
@patch("src.vector_store._load_embeddings")
def test_as_retriever_uses_configured_k(mock_embeddings, mock_faiss, settings, sample_docs):
    mock_embeddings.return_value = MagicMock()
    mock_store = MagicMock()
    mock_faiss.from_documents.return_value = mock_store

    settings.retriever_top_k = 7
    manager = VectorStoreManager(settings)
    manager.build(sample_docs)
    manager.as_retriever()

    mock_store.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={"k": 7}
    )
