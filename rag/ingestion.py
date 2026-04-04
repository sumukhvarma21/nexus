"""
RAG ingestion pipeline.

Flow: file → load → parent chunks → child chunks → embed children → store both
      (child chunks used for retrieval, parent chunks returned to LLM)
"""

import os
import uuid
from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _get_loader(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path)
    elif suffix == ".txt":
        return TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .txt")


def ingest_file(file_path: str) -> dict:
    """
    Ingest a file into ChromaDB using parent-child chunking.

    - Child chunks (256 tokens): stored and used for retrieval
    - Parent chunks (1024 tokens): stored with same doc_id, returned to LLM for context

    Returns metadata about the ingestion.
    """
    os.makedirs(settings.uploads_dir, exist_ok=True)
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)

    loader = _get_loader(file_path)
    raw_docs = loader.load()

    # Parent splitter — larger chunks sent to the LLM
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=100,
    )

    # Child splitter — smaller chunks used for retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=20,
    )

    parent_docs = parent_splitter.split_documents(raw_docs)

    child_docs: list[Document] = []
    for parent in parent_docs:
        parent_id = str(uuid.uuid4())
        parent.metadata["doc_id"] = parent_id
        parent.metadata["chunk_type"] = "parent"
        parent.metadata["source_file"] = Path(file_path).name

        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = parent_id
            child.metadata["chunk_type"] = "child"
            child.metadata["source_file"] = Path(file_path).name
        child_docs.extend(children)

    embeddings = _get_embeddings()

    # Store children (for retrieval)
    child_store = Chroma(
        collection_name="child_chunks",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    child_store.add_documents(child_docs)

    # Store parents (for context — no embedding needed, just lookup by parent_id)
    parent_store = Chroma(
        collection_name="parent_chunks",
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    parent_store.add_documents(parent_docs)

    return {
        "file": Path(file_path).name,
        "parent_chunks": len(parent_docs),
        "child_chunks": len(child_docs),
        "source_pages": len(raw_docs),
    }
