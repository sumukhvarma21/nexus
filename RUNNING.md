# Running Nexus

## Prerequisites

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here        # optional, enables web search
LANGCHAIN_API_KEY=your_key_here     # optional, enables LangSmith tracing
```

---

## Start the backend (FastAPI)

```bash
source venv/bin/activate
uvicorn api.routes:app --reload --port 8000
```

---

## Start the UI (Chainlit)

In a second terminal:

```bash
source venv/bin/activate
chainlit run ui/app.py --port 8080
```

Open **http://localhost:8080** in your browser.

---

## Start the MCP server (optional)

For Claude Desktop or other MCP clients. In a third terminal:

```bash
source venv/bin/activate
python mcp_server/server.py
```

---

## Run tests

```bash
source venv/bin/activate
pytest tests/
```

Run a single test file:

```bash
pytest tests/test_orchestrator.py -v
```

---

## Ports summary

| Service | Port | URL |
|---------|------|-----|
| FastAPI backend | 8000 | http://localhost:8000 |
| Chainlit UI | 8080 | http://localhost:8080 |
| FastAPI docs | 8000 | http://localhost:8000/docs |
