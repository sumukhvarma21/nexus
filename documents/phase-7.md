# Phase 7 — AWS Serverless Deployment

**Goal:** Move everything to AWS. Serverless where possible. Production-grade configuration.

**Depends on:** [Phase 6](phase-6.md)

## AWS Architecture

```
                    CloudFront (CDN)
                          ↓
                    API Gateway (HTTP)
                    ↙              ↘
     Lambda (FastAPI via Mangum)   Lambda (MCP Server)
               ↓                           ↓
       ECS Fargate Tasks           Google MCP Servers
       (long-running LangGraph      (Gmail, GCal)
        workflows, async)
               ↓
      ┌─────────────────┐
      │  Qdrant on ECS  │   S3 (document storage)
      │  (vector store) │   ElastiCache Redis (sessions)
      └─────────────────┘   Secrets Manager (env vars)
                            CloudWatch (logs + traces)
```

**Why Lambda + ECS split:** Lambda has 15-minute timeout. LangGraph workflows with multiple agent hops + retrieval can exceed this. Short calls (health, ingest triggers) → Lambda. Long agentic workflows → ECS Fargate async.

## ChromaDB → Qdrant Migration

| | ChromaDB | Qdrant |
|---|---|---|
| Managed AWS | None | ECS or Qdrant Cloud |
| Filtering | Basic metadata | Advanced payload filtering |
| Multi-tenancy | Manual | Built-in collection isolation |
| Performance | Good | Excellent |

One-time migration script. Worth it for production.

## AWS Services Map

| AWS Service | Replaces | Purpose |
|---|---|---|
| API Gateway + Lambda | `uvicorn` local | Serverless HTTP |
| ECS Fargate | Long-running local processes | LangGraph async workflows |
| S3 | Local file storage | Document uploads |
| Qdrant on ECS | ChromaDB | Production vector store |
| ElastiCache Redis | In-memory session state | Distributed session memory |
| Secrets Manager | `.env` file | Secure config |
| CloudWatch | Print statements | Logging + tracing |
| ECR | Local Docker images | Container registry |

## Deliverables

- [ ] `Dockerfile` for FastAPI app
- [ ] Mangum adapter wiring FastAPI to Lambda
- [ ] `scripts/migrate_chroma_to_qdrant.py` — one-time migration
- [ ] CDK (Python) for all infrastructure
- [ ] GitHub Actions CI/CD: push → ECR → ECS deploy
- [ ] Environment-aware config: `config.py` switches local vs AWS automatically
- [ ] Redis session store replacing in-memory
- [ ] Full system running end-to-end on AWS
- [ ] Load test: 10 concurrent users, measure p95 latency

## Packages

```
mangum
boto3
qdrant-client
redis
```
