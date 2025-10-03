"""
RAG Chatbot (LlamaIndex + Chroma + Groq + Gradio)

What this does
--------------
- Loads documents from ./data, chunks them, builds embeddings, and stores them in Chroma (persistent).
- Serves a simple chat UI that runs retrieval (top-K) + LLM synthesis via Groq.
- Skips reranking in code; a plug-in point is shown below if you want to add it later.

Run:
  python main.py             # load/reuse existing index if present
  python main.py --rebuild   # force re-index from ./data

Env (.env):
  GROQ_API_KEY=...
  GROQ_MODEL=llama-3.1-8b-instant
  EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
  TOP_K=4
  CHROMA_DIR=./chroma_db
  CHROMA_COLLECTION=rag_collection
  DATA_DIR=./data
"""

import os
import sys
from pathlib import Path
from typing import List

# Keep tokenizer libs quiet
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from dotenv import load_dotenv
import gradio as gr
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


# -----------------------------
# Configuration & wiring
# -----------------------------
def load_settings():
    load_dotenv()  # reads .env if present

    cfg = {
        "groq_api_key": os.getenv("GROQ_API_KEY", "").strip(),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip(),
        "embed_model": os.getenv(
            "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        "top_k": int(os.getenv("TOP_K", "4")),
        "chroma_dir": os.getenv("CHROMA_DIR", "./chroma_db").strip(),
        "chroma_collection": os.getenv("CHROMA_COLLECTION", "rag_collection").strip(),
        "data_dir": os.getenv("DATA_DIR", "./data").strip(),
    }

    if not cfg["groq_api_key"]:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Set it in environment or .env file."
        )

    # Set LLM and embeddings globally for LlamaIndex APIs
    Settings.llm = Groq(model=cfg["groq_model"], api_key=cfg["groq_api_key"])
    Settings.embed_model = HuggingFaceEmbedding(model_name=cfg["embed_model"])

    return cfg


def init_vector_store(cfg):
    # Persistent Chroma client + collection
    client = chromadb.PersistentClient(path=cfg["chroma_dir"])
    collection = client.get_or_create_collection(cfg["chroma_collection"])
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    return client, collection, vector_store, storage_ctx


def collection_is_empty(collection) -> bool:
    try:
        return (collection.count() or 0) == 0
    except Exception:
        return True


def build_or_load_index(cfg, storage_ctx, collection, rebuild=False, client=None):
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    if rebuild or collection_is_empty(collection):
        print("[index] Building index from ./data ...")

        # For rebuild, delete existing collection to ensure clean state
        if rebuild and client is not None:
            collection_name = cfg["chroma_collection"]
            try:
                client.delete_collection(collection_name)
                print(f"[index] Deleted existing collection '{collection_name}' for clean rebuild")
            except Exception as e:
                print(f"[index] Note: Could not delete collection '{collection_name}' (may not exist): {e}")
            
            # Re-initialize the vector store after deletion
            client, collection, _, storage_ctx = init_vector_store(cfg)

        # SimpleDirectoryReader auto-detects formats; .txt/.md are easiest;
        # install PyMuPDF (pymupdf) if you want robust PDF parsing.
        docs = SimpleDirectoryReader(
            input_dir=str(data_dir), recursive=True, filename_as_id=True
        ).load_data()

        if not docs:
            print(
                "[index] No documents found in ./data. "
                "Add some .txt or .md files (PDFs optional) and re-run."
            )

        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_ctx, show_progress=True
        )
        print("[index] Done.")
    else:
        print("[index] Reusing existing Chroma collection.")
        # Build a "view" over the existing vector store
        index = VectorStoreIndex.from_vector_store(storage_ctx.vector_store)

    return index, storage_ctx, collection


# -----------------------------
# Optional: Reranker hook (not used in this minimal app)
# -----------------------------
# If/when you want to try reranking, uncomment below and pass node_postprocessors to as_query_engine.
# Requires: pip install llama-index-postprocessor-flag-embedding-reranker FlagEmbedding
# from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
# reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=4)


# -----------------------------
# Chat function
# -----------------------------
def format_sources(source_nodes) -> str:
    """Render a compact sources list from Response.source_nodes."""
    if not source_nodes:
        return ""
    seen = set()
    lines: List[str] = []
    for sn in source_nodes:
        # Compatible across LlamaIndex versions
        meta = {}
        try:
            meta = sn.node.metadata or {}
            score = getattr(sn, "score", None)
        except Exception:
            meta = getattr(sn, "metadata", {}) or {}
            score = getattr(sn, "score", None)

        src = (
            meta.get("file_path")
            or meta.get("file_name")
            or meta.get("source")
            or meta.get("document_id")
            or "unknown"
        )
        if src not in seen:
            seen.add(src)
            if score is not None:
                lines.append(f"- {src} (score={score:.3f})")
            else:
                lines.append(f"- {src}")
    return "\n".join(lines[:5])  # up to 5 unique sources


def make_chat_engine(index, top_k: int):
    # Use chat engine for conversation history support
    return index.as_chat_engine(
        similarity_top_k=top_k,
        response_mode="compact",
        chat_mode="best",  # Maintains conversation context
    )


def build_app(query_engine, cfg):
    title = "RAG Chat — LlamaIndex + Chroma + Groq"
    description = (
        "Ask questions about your local documents (in ./data). "
        "Run `python main.py --rebuild` after adding files to re-index."
    )

    def rag_chat(message, history):
        # Use chat engine which maintains conversation history
        if not hasattr(rag_chat, 'chat_engine'):
            # Initialize chat engine on first call
            rag_chat.chat_engine = query_engine

        resp = rag_chat.chat_engine.chat(message)
        text = str(resp)
        sources = format_sources(getattr(resp, "source_nodes", []))
        if sources:
            text += "\n\n**Sources**\n" + sources

        # Return in new Gradio messages format
        return [
            {"role": "user", "content": message},
            {"role": "assistant", "content": text}
        ]

    # Simple, one-function chat interface
    return gr.ChatInterface(
        fn=rag_chat,
        title=title,
        description=description,
        examples=None,
        theme=None,
        cache_examples=False,
        type="messages",
    )


def main():
    cfg = load_settings()
    client, collection, _, storage_ctx = init_vector_store(cfg)
    rebuild = "--rebuild" in sys.argv
    index, storage_ctx, collection = build_or_load_index(cfg, storage_ctx, collection, rebuild=rebuild, client=client)
    chat_engine = make_chat_engine(index, top_k=cfg["top_k"])
    app = build_app(chat_engine, cfg)
    app.launch()


if __name__ == "__main__":
    main()
