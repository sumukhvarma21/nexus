import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from rag.ingestion import ingest_file
from rag.retrieval import retrieve

app = FastAPI(title="Nexus", version="0.1.0")

os.makedirs(settings.uploads_dir, exist_ok=True)


class QueryRequest(BaseModel):
    query: str


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
