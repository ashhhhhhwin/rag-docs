
import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_BASE = f"{API_URL}/api/v1"

st.set_page_config(
    page_title="Chat With Your Docs",
    page_icon="📚",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []


# -----------------------------
# API functions
# -----------------------------

def get_documents():
    try:
        response = httpx.get(
            f"{API_BASE}/documents",
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def upload_document(uploaded_file):
    try:
        response = httpx.post(
            f"{API_BASE}/documents/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", "Upload failed.")
        except ValueError:
            detail = "Upload failed."
        st.error(f"Upload error: {detail}")
        return None

    except httpx.HTTPError as exc:
        st.error(f"Could not connect to the backend: {exc}")
        return None


def ask_question(question):
    try:
        response = httpx.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get(
                "detail", "Unable to generate an answer."
            )
        except ValueError:
            detail = "Unable to generate an answer."
        st.error(f"Answer generation failed: {detail}")
        return None

    except httpx.HTTPError as exc:
        st.error(f"Could not connect to the backend: {exc}")
        return None


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.title("📚 Document Library")
    st.caption("Upload documents and build your knowledge base.")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md"],
        help="Maximum file size: 10 MB",
    )

    if st.button(
        "Upload and Index",
        disabled=uploaded_file is None,
        use_container_width=True,
    ):
        with st.spinner("Extracting text and indexing document..."):
            result = upload_document(uploaded_file)

        if result:
            st.success(
                f"{result['filename']} indexed successfully. "
                f"{result['chunks_created']} chunks created."
            )
            st.session_state.documents = get_documents() or []

    st.divider()

    st.subheader("Indexed Documents")

    if st.button("Refresh Documents", use_container_width=True):
        st.session_state.documents = get_documents() or []

    if not st.session_state.documents:
        documents = get_documents()
        if documents is not None:
            st.session_state.documents = documents

    for document in st.session_state.documents:
        st.markdown(f"📄 **{document['filename']}**")

    st.divider()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by FastAPI, ChromaDB and OpenRouter.")


# -----------------------------
# Main chat interface
# -----------------------------

st.title("💬 Chat With Your Docs")

st.markdown(
    "Ask questions about your uploaded documents. "
    "Answers are generated using retrieved document context."
)

if not st.session_state.documents:
    st.info(
        "Upload a PDF, TXT or Markdown document from the sidebar "
        "to get started."
    )

# Display previous messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 View source references"):
                for source in message["sources"]:
                    page = (
                        f" · Page {source['page']}"
                        if source["page"] is not None
                        else ""
                    )
                    st.markdown(
                        f"**[{source['citation_id']}]** "
                        f"{source['source']}{page}"
                    )

            st.caption(f"Model: {message['model']}")

# Chat input

question = st.chat_input(
    "Ask something about your documents...",
    max_chars=1000,
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = ask_question(question)

        if result:
            answer = result["answer"]
            sources = result["sources"]

            st.markdown(answer)

            with st.expander("📚 View source references"):
                for source in sources:
                    page = (
                        f" · Page {source['page']}"
                        if source["page"] is not None
                        else ""
                    )
                    st.markdown(
                        f"**[{source['citation_id']}]** "
                        f"{source['source']}{page}"
                    )

            st.caption(f"Model: {result['model']}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "model": result["model"],
                }
            )
        else:
            st.warning(
                "I couldn't generate an answer. "
                "Please check the backend or try again."
            )
