"""
Conversational RAG chain.

Built with LangChain's LCEL primitives rather than the deprecated
`ConversationalRetrievalChain`: a history-aware retriever first reformulates
the question using chat context, then a stuff-documents chain generates a
grounded answer with explicit source attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq

from src.config import Settings
from src.exceptions import LLMConfigurationError
from src.logging_config import get_logger

logger = get_logger(__name__)

_CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, rephrase the "
            "follow-up into a standalone question that can be understood without the "
            "chat history. Do not answer it — only rephrase it. If it is already "
            "standalone, return it unchanged.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise, professional document assistant. Answer the user's "
            "question using ONLY the context below. If the answer isn't in the "
            "context, say you don't have enough information rather than guessing. "
            "Be concise and cite which part of the document supports your answer "
            "when relevant.\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


@dataclass
class RAGResponse:
    answer: str
    sources: list[Document] = field(default_factory=list)


class RAGChatbot:
    """Orchestrates the LLM, retriever, and prompts into a single query interface."""

    def __init__(self, settings: Settings, retriever: VectorStoreRetriever):
        if not settings.groq_api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is not set. Provide it via the sidebar or a .env file."
            )

        self._settings = settings
        self._retriever = retriever
        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    @staticmethod
    def _to_lc_history(history: list[tuple[str, str]]) -> list:
        messages = []
        for user_msg, ai_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=ai_msg))
        return messages

    def _condense_question(self, question: str, history: list) -> str:
        if not history:
            return question
        chain = _CONDENSE_QUESTION_PROMPT | self._llm
        result = chain.invoke({"chat_history": history, "input": question})
        return result.content.strip()

    def ask(self, question: str, chat_history: list[tuple[str, str]] | None = None) -> RAGResponse:
        """Answer a question grounded in the indexed documents."""
        history = self._to_lc_history(chat_history or [])
        standalone_question = self._condense_question(question, history)

        docs = self._retriever.invoke(standalone_question)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
        )

        chain = _ANSWER_PROMPT | self._llm
        result = chain.invoke(
            {"context": context, "chat_history": history, "input": standalone_question}
        )

        logger.info("Answered question using %d retrieved chunk(s)", len(docs))
        return RAGResponse(answer=result.content, sources=docs)
