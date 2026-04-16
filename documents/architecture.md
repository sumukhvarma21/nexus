## Application Workflow

End-to-end query lifecycle inside the multi-agent system. Shows routing, per-agent flow (including the full RAG pipeline), and inline memory reads/writes.

```mermaid
flowchart TD
    User([User]) -->|query + session_id| API[FastAPI Backend<br/>api/routes.py]
    MCPClient([LLM Client<br/>Claude Desktop]) -->|tool call| MCP[FastMCP Server<br/>mcp_server/server.py]
    MCP -->|HTTP| API

    API --> LoadMem[Load Short-Term State<br/>LangGraph checkpoint]
    LoadMem --> LoadFacts[Load User Facts<br/>SQLite]
    LoadFacts --> Supervisor{Supervisor Node<br/>Gemini Flash 2.5<br/>structured output}

    Supervisor -->|agent: rag_agent| RAG[RAG Agent]
    Supervisor -->|agent: web_search_agent| Web[Web Search Agent]
    Supervisor -->|agent: email_calendar_agent| EmailCal[Email/Calendar Agent]

    subgraph RAGFlow[RAG Agent Pipeline]
        direction TB
        R1[Query Decomposition<br/>sub-questions] --> R2[HyDE<br/>generate hypothetical answer]
        R2 --> R3[Embed hypothetical<br/>BAAI/bge-base-en-v1.5]
        R3 --> R4[Bi-Encoder ANN Search<br/>top-20 child chunks]
        R4 --> R5[Cross-Encoder Rerank<br/>ms-marco-MiniLM top-5]
        R5 --> R6[Parent Chunk Lookup<br/>256→1024 tokens]
        R6 --> R7[Episodic Memory Retrieval<br/>similar past Q&A]
        R7 --> R8[Assemble Context]
        R8 --> R9[Generate Answer<br/>Gemini Flash 2.5]
        R9 --> R10{Faithfulness Check<br/>claims ↔ context}
        R10 -->|fail| R2
        R10 -->|pass| R11[Answer + Citations]
    end

    RAG --> RAGFlow
    RAGFlow --> Synth

    subgraph WebFlow[Web Search Agent]
        direction TB
        W1[Tavily API Search] --> W2[Rerank Results] --> W3[Summarize]
    end
    Web --> WebFlow
    WebFlow --> Synth

    subgraph EmailFlow[Email/Calendar Agent]
        direction TB
        E1[Parse Intent] --> E2[Gmail / Calendar API] --> E3[Format Response]
    end
    EmailCal --> EmailFlow
    EmailFlow --> Synth

    Synth[Synthesizer Node<br/>final answer] --> SaveShort[Update Short-Term State<br/>summarization buffer]
    SaveShort --> SaveEpisodic[Write Episodic Memory<br/>ChromaDB long-term]
    SaveEpisodic --> ExtractFacts[Extract User Facts<br/>LLM → SQLite]
    ExtractFacts --> Response([Response to User])

    ChromaChild[(ChromaDB<br/>child chunks)]
    ChromaParent[(Parent Store<br/>1024-tok chunks)]
    ChromaEpisodic[(ChromaDB<br/>episodic memory)]
    SQLiteFacts[(SQLite<br/>user facts)]
    State[(LangGraph State<br/>short-term)]

    R4 -.reads.-> ChromaChild
    R6 -.reads.-> ChromaParent
    R7 -.reads.-> ChromaEpisodic
    LoadMem -.reads.-> State
    LoadFacts -.reads.-> SQLiteFacts
    SaveShort -.writes.-> State
    SaveEpisodic -.writes.-> ChromaEpisodic
    ExtractFacts -.writes.-> SQLiteFacts

    classDef agent fill:#e1f5ff,stroke:#0288d1
    classDef memory fill:#fff4e1,stroke:#f57c00
    classDef llm fill:#f3e5f5,stroke:#7b1fa2
    class RAG,Web,EmailCal,Supervisor,Synth agent
    class ChromaChild,ChromaParent,ChromaEpisodic,SQLiteFacts,State memory
    class R2,R9,R10,ExtractFacts,Supervisor llm
```

---

## Production Architecture — AWS Serverless

Query-time and ingestion-time paths. Lambda for short calls, ECS Fargate for long LangGraph workflows, SQS for async fan-out, Qdrant Cloud external to AWS.

```mermaid
flowchart TB
    subgraph Client[Clients]
        Browser([Browser])
        ClaudeDesktop([Claude Desktop / LLM Client])
    end

    subgraph Edge[Edge / CDN]
        CF[CloudFront]
        S3UI[(S3<br/>Chainlit UI static)]
    end

    Browser --> CF --> S3UI
    CF -->|/api/*| APIGW[API Gateway<br/>REST]
    ClaudeDesktop -->|MCP protocol| APIGW

    subgraph Compute[Compute Layer]
        MCPLambda[Lambda<br/>FastMCP handler]
        APILambda[Lambda<br/>FastAPI router<br/>short calls]
        Orchestrator[ECS Fargate<br/>LangGraph orchestrator<br/>long workflows]
        Ingestor[Lambda<br/>Ingestion worker]
        FactExtractor[Lambda<br/>User fact extractor]
    end

    APIGW --> MCPLambda
    APIGW --> APILambda
    APILambda -->|invoke async| Orchestrator

    subgraph State[State & Memory]
        Qdrant[(Qdrant Cloud<br/>chunks + episodic)]
        DDBSession[(DynamoDB<br/>session state)]
        DDBFacts[(DynamoDB<br/>user facts)]
        DDBCheckpoint[(DynamoDB<br/>LangGraph checkpoints)]
    end

    Orchestrator <-->|retrieve / write| Qdrant
    Orchestrator <-->|short-term state| DDBSession
    Orchestrator <-->|checkpoints| DDBCheckpoint
    Orchestrator <-->|user facts| DDBFacts

    subgraph Async[Async Pipeline]
        IngestQueue[[SQS<br/>ingestion queue]]
        MemoryQueue[[SQS<br/>memory write queue]]
        FactQueue[[SQS<br/>fact extraction queue]]
        DLQ[[SQS DLQ]]
    end

    subgraph Ingestion[Document Ingestion]
        S3Docs[(S3<br/>raw documents)]
        S3Docs -->|ObjectCreated event| IngestQueue
        IngestQueue --> Ingestor
        Ingestor -->|chunk + embed| Qdrant
    end

    Orchestrator -->|post-response| MemoryQueue
    Orchestrator -->|post-response| FactQueue
    MemoryQueue --> MemWriter[Lambda<br/>episodic memory writer]
    MemWriter --> Qdrant
    FactQueue --> FactExtractor
    FactExtractor --> DDBFacts

    IngestQueue -.failures.-> DLQ
    MemoryQueue -.failures.-> DLQ
    FactQueue -.failures.-> DLQ

    subgraph External[External Services]
        Gemini[Gemini Flash 2.5 API]
        Tavily[Tavily Search API]
        Gmail[Gmail / Google Calendar API]
    end

    Orchestrator --> Gemini
    Orchestrator --> Tavily
    Orchestrator --> Gmail

    subgraph Observability[Observability]
        CW[CloudWatch<br/>logs + metrics + alarms]
        LS[LangSmith<br/>LLM traces]
    end

    MCPLambda -.-> CW
    APILambda -.-> CW
    Orchestrator -.-> CW
    Ingestor -.-> CW
    Orchestrator -.-> LS

    Secrets[[Secrets Manager<br/>API keys]]
    APILambda -.-> Secrets
    Orchestrator -.-> Secrets
    Ingestor -.-> Secrets

    classDef lambda fill:#fff4e1,stroke:#f57c00
    classDef storage fill:#e8f5e9,stroke:#388e3c
    classDef queue fill:#fce4ec,stroke:#c2185b
    classDef external fill:#ede7f6,stroke:#512da8
    class MCPLambda,APILambda,Ingestor,FactExtractor,MemWriter lambda
    class Qdrant,DDBSession,DDBFacts,DDBCheckpoint,S3Docs,S3UI storage
    class IngestQueue,MemoryQueue,FactQueue,DLQ queue
    class Gemini,Tavily,Gmail external
```

---

## Notes

- **Why Lambda + Fargate split:** Lambda caps at 15 min and has cold-start latency painful for short requests; LangGraph multi-agent workflows with iterative retrieval can exceed that. Fargate runs the long-lived orchestrator; Lambda fronts it for short synchronous paths (tool discovery, health, cached reads).
- **Why SQS between orchestrator and memory writers:** episodic memory + fact extraction shouldn't block the user-facing response. Fire-and-forget post-response.
- **Why DynamoDB over RDS:** session state and user facts are key-lookup workloads (by session_id, user_id). DynamoDB is serverless, sub-ms, no connection pooling pain from Lambda.
- **Qdrant Cloud (not self-hosted):** avoids running stateful vector DB infra in AWS. Trade-off: cross-cloud latency, external dependency.
