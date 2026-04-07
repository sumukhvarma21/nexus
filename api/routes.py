import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from rag.ingestion import ingest_file
from rag.retrieval import retrieve, retrieve_with_hyde, retrieve_multi_step, retrieve_iterative
from agents.orchestrator import run_chat

app = FastAPI(title="Nexus", version="0.1.0")

os.makedirs(settings.uploads_dir, exist_ok=True)


class QueryRequest(BaseModel):
    query: str


class ChatRequest(BaseModel):
    query: str
    session_id: str = ""


class AdvancedQueryRequest(BaseModel):
    query: str
    mode: str = "multi_step"  # standard | hyde | multi_step | iterative


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files supported")

    dest = os.path.join(settings.uploads_dir, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingest_file(dest)
    return JSONResponse(content=result)


@app.post("/query")
async def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chunks = retrieve(request.query)

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage

    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    sources = list({c["source_file"] for c in chunks})

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context doesn't contain the answer, say so."
            )
        ),
        HumanMessage(
            content=f"Context:\n{context}\n\nQuestion: {request.query}"
        ),
    ]

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
    }


@app.post("/query/advanced")
async def query_advanced(request: AdvancedQueryRequest):
    """
    Multi-step RAG query endpoint.

    Modes:
      - standard:   baseline two-stage retrieval (same as /query)
      - hyde:       HyDE — embed hypothetical answer instead of raw query
      - multi_step: decompose query → retrieve per sub-question → merge
      - iterative:  retrieve → check sufficiency → refine → repeat (max 3 hops)
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    valid_modes = {"standard", "hyde", "multi_step", "iterative"}
    if request.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{request.mode}'. Choose from: {sorted(valid_modes)}",
        )

    hops_used = None

    if request.mode == "standard":
        chunks = retrieve(request.query)
    elif request.mode == "hyde":
        chunks = retrieve_with_hyde(request.query)
    elif request.mode == "multi_step":
        chunks = retrieve_multi_step(request.query)
    elif request.mode == "iterative":
        chunks, hops_used = retrieve_iterative(request.query)

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": [], "mode": request.mode}

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage

    context = "\n\n---\n\n".join(c["content"] for c in chunks)
    sources = list({c["source_file"] for c in chunks})

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the context doesn't contain the answer, say so."
            )
        ),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.query}"),
    ]

    response = llm.invoke(messages)

    result = {
        "answer": response.content,
        "sources": sources,
        "chunks_used": len(chunks),
        "mode": request.mode,
    }
    if hops_used is not None:
        result["hops_used"] = hops_used

    return result


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Multi-agent chat endpoint (Phase 3).

    The supervisor routes the query to the appropriate agent:
      - rag_agent        → queries about uploaded documents
      - web_search_agent → queries needing current/external information

    Returns the answer plus metadata about which agent was used.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = run_chat(request.query, session_id=request.session_id)

    return {
        "answer": result["answer"],
        "agent_called": result["agent_called"],
        "sources_used": len(result["retrieved_context"]) + len(result["web_results"]),
        "session_id": result["session_id"],
    }
