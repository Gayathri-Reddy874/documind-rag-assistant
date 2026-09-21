"""
Enterprise Document Intelligence Assistant
Streamlit entry point.

Run with: streamlit run app.py
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import streamlit as st

from src.config import get_settings
from src.document_processor import DocumentProcessor, UploadedFile
from src.exceptions import DocumentAssistantError, LLMConfigurationError
from src.logging_config import configure_logging, get_logger
from src.rag_chain import RAGChatbot
from src.vector_store import VectorStoreManager

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)

st.set_page_config(
    page_title=settings.app_name,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 2rem; max-width: 1100px;}

        .app-header {
            display: flex; align-items: center; gap: 0.75rem;
            margin-bottom: 0.25rem;
        }
        .app-header h1 {margin: 0; font-size: 1.75rem; font-weight: 700;}
        .app-subtitle {color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem;}

        .status-pill {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.25rem 0.7rem; border-radius: 999px;
            font-size: 0.8rem; font-weight: 600;
        }
        .status-ready {background: #ecfdf5; color: #047857;}
        .status-pending {background: #fef3c7; color: #92400e;}

        .source-chip {
            display: inline-block; padding: 0.15rem 0.55rem; margin: 0.15rem;
            border-radius: 6px; background: #f3f4f6; color: #374151;
            font-size: 0.78rem; border: 1px solid #e5e7eb;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 12px; padding: 0.5rem 0.25rem;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"?}
if "vector_manager" not in st.session_state:
    st.session_state.vector_manager = VectorStoreManager(settings)
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

processor = DocumentProcessor(settings)


def _chat_history_pairs() -> list[tuple[str, str]]:
    """Turn the flat message list into (user, assistant) pairs for the chain."""
    pairs, pending_user = [], None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            pending_user = msg["content"]
        elif msg["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, msg["content"]))
            pending_user = None
    return pairs


# --------------------------------------------------------------------------- #
# Sidebar — configuration & document management
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=settings.groq_api_key,
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    if api_key_input:
        settings.groq_api_key = api_key_input

    with st.expander("Advanced settings"):
        settings.groq_model = st.selectbox(
            "Model",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
            ],
            index=0,
            help=(
                "Model availability on Groq changes over time. If a model returns "
                "'model_not_found', check https://console.groq.com/docs/models for "
                "the current list of models available on your plan."
            ),
        )
        settings.llm_temperature = st.slider("Temperature", 0.0, 1.0, settings.llm_temperature, 0.05)
        settings.retriever_top_k = st.slider("Chunks retrieved (k)", 1, 10, settings.retriever_top_k)

    st.divider()
    st.markdown("### 📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=[ext.lstrip(".") for ext in settings.allowed_extensions],
        accept_multiple_files=True,
    )

    process_clicked = st.button("🚀 Process documents", use_container_width=True, type="primary")

    if process_clicked:
        if not settings.groq_api_key:
            st.error("Please enter your Groq API key first.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            try:
                with st.spinner("Reading, chunking, and embedding documents..."):
                    files = [UploadedFile(name=f.name, data=f.read()) for f in uploaded_files]
                    chunks = processor.process(files)
                    st.session_state.vector_manager.build(chunks)
                    st.session_state.chatbot = RAGChatbot(
                        settings, st.session_state.vector_manager.as_retriever()
                    )
                    st.session_state.indexed_files = [f.name for f in files]
                    st.session_state.messages = []
                st.success(f"Indexed {len(files)} document(s) into {len(chunks)} chunks.")
            except DocumentAssistantError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 — surface unexpected errors to the user
                logger.exception("Unexpected error while processing documents")
                st.error(f"Something went wrong: {exc}")

    if st.session_state.indexed_files:
        st.markdown("**Indexed files:**")
        for name in st.session_state.indexed_files:
            st.markdown(f'<span class="source-chip">📄 {name}</span>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Session `{st.session_state.session_id}` · {datetime.now():%Y-%m-%d %H:%M}")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
is_ready = st.session_state.vector_manager.is_ready
status_html = (
    '<span class="status-pill status-ready">🟢 Ready</span>'
    if is_ready
    else '<span class="status-pill status-pending">🟡 Awaiting documents</span>'
)

st.markdown(
    f"""
    <div class="app-header">
        <h1>📚 {settings.app_name}</h1>
        {status_html}
    </div>
    <div class="app-subtitle">
        Retrieval-augmented Q&A over your own documents, powered by Groq's LPU inference.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Chat history
# --------------------------------------------------------------------------- #
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source chunk(s)"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**{i}. {src.metadata.get('source', 'unknown')}**")
                    st.caption(src.page_content[:400] + ("…" if len(src.page_content) > 400 else ""))

# --------------------------------------------------------------------------- #
# Chat input
# --------------------------------------------------------------------------- #
prompt = st.chat_input(
    "Ask a question about your document..." if is_ready else "Upload and process a document to begin"
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not is_ready:
            answer = "⚠️ Please upload and process a document first."
            st.warning(answer)
            sources = []
        else:
            try:
                with st.spinner("Thinking..."):
                    if st.session_state.chatbot is None:
                        st.session_state.chatbot = RAGChatbot(
                            settings, st.session_state.vector_manager.as_retriever()
                        )
                    t0 = time.time()
                    response = st.session_state.chatbot.ask(prompt, _chat_history_pairs())
                    elapsed = time.time() - t0
                answer = response.answer
                sources = response.sources
                st.markdown(answer)
                st.caption(f"⏱ {elapsed:.1f}s · {len(sources)} chunk(s) retrieved")
                if sources:
                    with st.expander(f"📎 {len(sources)} source chunk(s)"):
                        for i, src in enumerate(sources, start=1):
                            st.markdown(f"**{i}. {src.metadata.get('source', 'unknown')}**")
                            st.caption(
                                src.page_content[:400] + ("…" if len(src.page_content) > 400 else "")
                            )
            except LLMConfigurationError as exc:
                answer = f"⚠️ {exc}"
                st.error(answer)
                sources = []
            except DocumentAssistantError as exc:
                answer = f"⚠️ {exc}"
                st.error(answer)
                sources = []
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error while answering question")
                answer = f"⚠️ Something went wrong: {exc}"
                st.error(answer)
                sources = []

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
