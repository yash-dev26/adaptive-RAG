# Adaptive RAG

A production-oriented Adaptive Retrieval-Augmented Generation (RAG) system built with FastAPI, LangGraph, Qdrant, Redis, React, OpenAI, and Groq.

The system uses a multi-provider inference architecture with adaptive routing, semantic caching, retrieval evaluation, query rewriting, reranking, and cost-optimized orchestration.

Unlike traditional “retrieve → generate” pipelines, this project dynamically decides when retrieval is necessary, rewrites ambiguous queries, evaluates retrieval quality, reranks context, and falls back to a pure LLM response when retrieval is not beneficial.

> This project demonstrates production-oriented AI engineering patterns including adaptive retrieval, graph-based orchestration, semantic caching, multi-provider inference routing, and hybrid LLM architectures.


`NOTE : Reranking (disabled on low-memory deployments)`

![Python](https://img.shields.io/badge/-Python-14161A?style=flat-square&logo=python&logoColor=3776AB)
![LangGraph](https://img.shields.io/badge/-LangGraph-14161A?style=flat-square&logo=langchain&logoColor=1C3C3C)
![FastAPI](https://img.shields.io/badge/-FastAPI-14161A?style=flat-square&logo=fastapi&logoColor=009688)
![Qdrant](https://img.shields.io/badge/-Qdrant-14161A?style=flat-square&logo=qdrant&logoColor=DC244C)
![Redis](https://img.shields.io/badge/-Redis-14161A?style=flat-square&logo=redis&logoColor=FF4438)
![MongoDB](https://img.shields.io/badge/-MongoDB-14161A?style=flat-square&logo=mongodb&logoColor=47A248)
![React](https://img.shields.io/badge/-React-14161A?style=flat-square&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/-Vite-14161A?style=flat-square&logo=vite&logoColor=646CFF)
![TailwindCSS](https://img.shields.io/badge/-Tailwind-14161A?style=flat-square&logo=tailwindcss&logoColor=06B6D4)
![OpenAI](https://img.shields.io/badge/-OpenAI-14161A?style=flat-square&logo=openai&logoColor=FFFFFF)
![Groq](https://img.shields.io/badge/-Groq-14161A?style=flat-square&logo=groq&logoColor=F55036)
![Clerk](https://img.shields.io/badge/-Clerk-14161A?style=flat-square&logo=clerk&logoColor=6C47FF)
---

# Overview

This project implements an adaptive RAG pipeline capable of:

- Intelligent query routing
- Single-query and multi-query rewriting
- Reciprocal Rank Fusion (RRF)
- Context reranking
- Semantic and response caching
- Streaming graph execution
- Document-scoped retrieval
- Authentication and multi-user isolation
- Duplicate ingestion detection
- Multi-provider LLM orchestration
- Cost-optimized inference routing
- Provider abstraction layer
- Configurable model routing
- LLM fallback handling

The system supports PDF ingestion, semantic search, and conversational querying over uploaded documents.

---

# System Architecture

```text
                           ┌─────────────────────┐
                           │     React Client    │
                           │  (Vite + Clerk UI)  │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │     FastAPI API     │
                           │  Auth + Streaming   │
                           └──────────┬──────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
        ┌─────────────────┐                    ┌─────────────────┐
        │  Ingestion Flow │                    │   Chat Pipeline │
        └─────────────────┘                    └─────────────────┘
                 │                                         │
                 ▼                                         ▼
      ┌─────────────────────┐                 ┌─────────────────────┐
      │ PDF → Markdown      │                 │ LangGraph Workflow  │
      │ Chunking Pipeline   │                 │ Adaptive Routing    │
      └─────────────────────┘                 └─────────────────────┘
                 │                                         │
                 ▼                                         ▼
      ┌─────────────────────┐                 ┌─────────────────────┐
      │ Embedding Generation│                 │ Query Rewriting     │
      │ OpenAI Embeddings   │                 │ Retrieval Eval      │
      └─────────────────────┘                 │ (Groq Llama 3.3)    │
                 │                            └─────────────────────┘
                 ▼                                         │
      ┌─────────────────────┐                 ┌─────────────────────┐
      │      Qdrant         │◄────────────────│ Retrieval           │
      │ Vector Database     │                 │ + Reranking         │
      └─────────────────────┘                 └─────────────────────┘
                 ▲                                         │
                 │                                         ▼
      ┌─────────────────────┐                 ┌─────────────────────┐
      │ Redis Cache Layer   │                 │ Final Generation    │
      │ Embeddings/Response │                 │ OpenAI GPT-4.1-mini │
      └─────────────────────┘                 └─────────────────────┘


                               ┌─────────────────────────┐
                               │ Provider Abstraction    │
                               │ Centralized LLM Layer   │
                               └──────────┬──────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
                ┌─────────────────┐               ┌─────────────────┐
                │     OpenAI      │               │      Groq       │
                │ Embeddings      │               │ Fast Rewrites   │
                │ Final Responses │               │ Retrieval Eval  │
                └─────────────────┘               └─────────────────┘
```

---

# Adaptive RAG Workflow

```text
User Query
    │
    ▼
Pre-Retrieval Planner
    │
    ├── LLM Only
    │
    └── RAG Pipeline
            │
            ▼
    Query Rewriting
      ├── Single Rewrite
      └── Multi Rewrite
            │
            ▼
        Retrieval
            │
            ▼
   Retrieval Evaluation
            │
    ┌───────┼────────┐
    ▼       ▼        ▼
Generate  Rewrite   Fallback
            │
            ▼
        Reranking
            │
            ▼
      Final Response
```

---

# Multi-Provider LLM Architecture

The system uses a provider abstraction layer to support multiple LLM providers while keeping graph nodes provider-agnostic.

## Provider Responsibilities

| Task | Provider |
|---|---|
| Query Rewriting | Groq |
| Retrieval Evaluation | Groq |
| Final Response Generation | OpenAI |
| Embeddings | OpenAI |

## Why Multi-Provider?

This architecture improves:

- Cost efficiency
- Latency
- Provider flexibility
- Reliability
- Vendor portability

Fast orchestration tasks such as rewriting and evaluation are routed through Groq, while higher-quality final answer generation remains on OpenAI.

## Provider Abstraction Layer

All LLM calls are routed through a centralized provider service:

```text
Graph Nodes
    ↓
LLM Provider Layer
    ↓
┌───────────────┬───────────────┐
│    OpenAI     │     Groq      │
└───────────────┴───────────────┘
```

This makes the system easily extensible to additional providers such as Anthropic, Gemini, Together AI, or local models.

---

# Cost Optimization Strategy

The system minimizes inference costs through adaptive provider routing.

## Routing Strategy

| Operation | Strategy |
|---|---|
| Embeddings | OpenAI |
| Rewrite Tasks | Groq |
| Evaluator Tasks | Groq |
| Final Responses | OpenAI |
| Reranking | Local Cross Encoder |

## Why This Matters

Query rewriting and evaluation are high-frequency orchestration tasks that benefit more from low latency and low cost than maximum reasoning quality.

Final response generation is routed to OpenAI to maximize answer quality and response consistency.

This hybrid architecture significantly reduces operational costs while maintaining high-quality responses.

---

# How This Differs From Naive RAG

## Traditional RAG

Most beginner RAG systems follow a simple flow:

```text
User Query
   ↓
Vector Search
   ↓
LLM Generation
```

This works for simple cases, but breaks down when:

- Queries are ambiguous
- Retrieval quality is poor
- Documents are noisy
- Context ranking is inaccurate
- Retrieval is unnecessary
- Similar queries repeatedly hit the model

---

## Adaptive RAG

This system introduces decision-making between each stage.

### Key Improvements

| Feature | Naive RAG | Adaptive RAG |
|---|---|---|
| Query Planning | ❌ | ✅ |
| Query Rewriting | ❌ | ✅ |
| Multi-Query Retrieval | ❌ | ✅ |
| Retrieval Evaluation | ❌ | ✅ |
| Reranking | ❌ | ✅ |
| Semantic Cache | ❌ | ✅ |
| Response Cache | ❌ | ✅ |
| LLM Fallback | ❌ | ✅ |
| Streaming Pipeline | ❌ | ✅ |
| Multi-User Isolation | ❌ | ✅ |

---

# Query Rewriting

The system dynamically improves user queries before retrieval.

## Single Rewrite

Used for ambiguous or underspecified queries.

Example:

```text
Original:
"What does it say about caching?"

Rewritten:
"What does the uploaded document say about semantic caching?"
```

---

## Multi Rewrite

Generates multiple semantically related queries to improve recall.

Example:

```text
[
  "Explain semantic caching",
  "How does the response cache work?",
  "Describe Redis caching in the system"
]
```

The results are merged using Reciprocal Rank Fusion (RRF).

---

# Retrieval & Reranking

The retrieval system uses:

- OpenAI embeddings
- Qdrant vector search
- Payload filtering
- Reciprocal Rank Fusion
- Cross-encoder reranking

### Retrieval Pipeline

```text
Query
  ↓
Embedding Generation
  ↓
Qdrant Similarity Search
  ↓
Reciprocal Rank Fusion
  ↓
Cross Encoder Reranking
  ↓
Top Context Selection
```

---

# Caching System

The project includes multiple cache layers to reduce latency and API costs.

## 1. Embedding Cache

Stores embeddings in Redis to avoid recomputing vectors.

```text
Text → SHA256 → Redis
```

---

## 2. Response Cache

Caches exact user query responses.

```text
(user_id + file_id + query) → response
```

---

## 3. Semantic Cache

Caches semantically similar queries using vector similarity.

Instead of exact string matching:

```text
"What is semantic caching?"
```

and

```text
"Explain the semantic cache system"
```

can reuse the same cached response.

---

# Authentication & Security

Authentication is implemented using Clerk JWT verification.

## Features

- Protected API routes
- User-scoped document retrieval
- JWT validation
- Multi-user isolation
- Secure file ownership filtering

Every retrieval query is filtered by:

```text
user_id
file_id
```

This prevents users from accessing other users’ embeddings or documents.

---

# Rate Limiting & Cost Optimization

The system reduces unnecessary model usage through:

- Semantic caching
- Exact response caching
- Retrieval skipping
- Conditional reranking
- Duplicate document detection
- Query planning
- Limited rewrite attempts

### Optimization Examples

| Optimization | Benefit |
|---|---|
| Semantic Cache | Reduces repeated LLM calls |
| Embedding Cache | Reduces embedding costs |
| Duplicate Detection | Avoids re-ingestion |
| Retrieval Planner | Skips unnecessary retrieval |
| Conditional Rerank | Saves compute |

---

# Streaming Execution

The frontend receives live pipeline updates through Server-Sent Events (SSE).

Example execution stream:

```text
Planning retrieval...
Rewriting query...
Searching vector store...
Evaluating results...
Reranking context...
Generating answer...
```

This improves transparency and debugging visibility.

---

# Folder Structure

```text
Backend/
├── app/
│   ├── agent/
│   │   └── graph/
│   │       ├── nodes/
│   │       └── routing/
│   ├── api/
│   │   └── v1/
│   │       └── routes/
│   ├── auth/
│   ├── cache/
│   ├── config/
│   ├── ingestion/
│   ├── repository/
│   ├── retrieval/
│   ├── schemas/
│   ├── service/
│   └── main.py

Frontend/
├── src/
│   ├── components/
│   │   ├── chat/
│   │   ├── landing/
│   │   └── ui/
│   ├── hooks/
│   ├── lib/
│   ├── pages/
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css
│   └── App.css
├── package.json
├── vite.config.js
└── index.html
```

---

# Tech Stack

## Backend

- FastAPI
- LangGraph
- OpenAI API
- Groq API
- Qdrant
- Redis
- MongoDB
- SentenceTransformers
- CrossEncoder Reranking

## Frontend

- React
- Vite
- TailwindCSS
- Clerk Authentication

---

# Supported Features

## Current Capabilities

- PDF ingestion
- Semantic search
- Conversational querying
- Adaptive retrieval
- Streaming execution
- Query rewriting
- Multi-query retrieval
- Semantic caching
- Context reranking
- User authentication
- Multi-user support
- Multi-provider inference
- Provider abstraction layer
- Cost-optimized orchestration

---

# Real-World Applications of RAG

RAG systems can be used for:

- Internal company knowledge assistants
- Legal document analysis
- Research copilots
- Technical documentation search
- Customer support systems
- Healthcare knowledge retrieval
- Financial report assistants
- Enterprise search systems
- AI tutoring systems
- Multi-document reasoning systems

---

# What This Project Demonstrates

Building this project demonstrates practical understanding of:

## AI Engineering

- RAG architecture
- Vector databases
- Embeddings
- Retrieval systems
- Prompt engineering
- LLM orchestration
- Query optimization
- Evaluation pipelines

---

## Backend Engineering

- FastAPI architecture
- Async workflows
- Streaming APIs
- Authentication systems
- Caching systems
- Distributed system design
- API design
- Stateful orchestration

---

## Production-Oriented Design

- Cost optimization
- Retrieval evaluation
- Scalable architecture
- Retry handling
- Cache invalidation
- Duplicate detection
- User isolation
- Graph-based execution

---

# Running The Project

## Backend

```bash
cd Backend

pip install -r requirements.txt

uvicorn app.main:app --reload
```

---

## Frontend

```bash
cd Frontend

npm install

npm run dev
```

---

# Environment Variables

```env
OPENAI_API_KEY=
GROQ_API_KEY=

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=

SEMANTIC_CACHE_COLLECTION_NAME=

MONGODB_URI=

CLERK_PEM_PUBLIC_KEY=
VITE_CLERK_PUBLISHABLE_KEY=
```

---

# Project Goals

This project was built to explore:

- Adaptive retrieval systems
- Intelligent query planning
- Cost-efficient LLM pipelines
- Production-ready RAG architecture
- Real-world AI infrastructure patterns

---
