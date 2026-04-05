from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"

    # Embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # RAG chunking
    child_chunk_size: int = 256
    parent_chunk_size: int = 1024
    retrieval_top_k: int = 20
    rerank_top_n: int = 5

    # Storage
    chroma_persist_dir: str = "./chroma_store"
    uploads_dir: str = "./uploads"

    # LangSmith (optional)
    langchain_api_key: str = ""
    langchain_tracing_v2: bool = False
    langchain_project: str = "nexus"

    # Tavily (Phase 3)
    tavily_api_key: str = ""


settings = Settings()
