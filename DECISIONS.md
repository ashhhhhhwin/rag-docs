# Technical Decisions

I wanted to maintain this document to capture the decisions I made while building this project, why I made them, and what I might revisit after evaluating the application.

## 1. LLM Provider – OpenRouter

I decided to use OpenRouter for LLM integration instead of downloading and running a local LLM.

The main reasons were:
- It provides access to multiple LLM providers through a common API.
- I could experiment with available free-tier models without setting up a local inference environment.
- It was relatively straightforward to integrate using HTTP requests.
- I can switch models through environment configuration without changing the application logic.

One limitation I noticed is that free-tier models can have strict rate limits and shared capacity. This influenced some of my other architectural decisions, especially around chunking.

## 2. Architecture Planning and AI Assistance

I used Claude during the initial architecture planning and project scaffolding.

I wanted to use AI assistance to accelerate the initial setup, but I also wanted to understand the components being implemented rather than blindly accepting generated code.

I have maintained a separate `AI_USAGE.md` file to document where AI tools were used during development.

## 3. Chunking Strategy – Recursive Splitting Instead of Fully LLM-Based Chunking

Initially, I was considering an entirely LLM-based chunking strategy, where an LLM would help identify meaningful semantic boundaries in documents.

However, I realised that this would introduce additional API calls, latency, cost, and rate-limit dependencies during ingestion.

I decided to start with recursive, separator-based chunking instead.

The implementation attempts to split content using the following hierarchy:

1. Paragraph boundaries (`\n\n`)
2. Line boundaries (`\n`)
3. Sentence boundaries (`. `)
4. Word boundaries (` `)
5. Character-based splitting as a fallback for unusually long text

This gives me a predictable baseline without requiring an LLM call for every document chunk.

I may revisit semantic or LLM-assisted chunking after building an evaluation dataset and comparing retrieval quality.

## 4. Chunk Overlap

I decided to apply overlap after splitting the document into chunks.

The reason is that I wanted to preserve some context between neighbouring chunks without making the overlap itself determine the initial splitting boundaries.

The implementation takes a portion of the previous chunk and adds it to the beginning of the next chunk, attempting to start at a word boundary.

This should help when an important sentence or explanation is split across two chunks.

One thing I want to validate through evaluation is whether the chosen overlap introduces too much redundant context.

## 5. Initial Chunk Size and Overlap

I started with the following configuration:

- Chunk size: 800 characters
- Chunk overlap: 120 characters

These are initial values, not values I consider optimal.

I chose them as a starting point to get the ingestion and retrieval pipeline working. I intend to evaluate different chunk sizes and overlap values against a small set of representative questions before settling on a better configuration.

Note: The current implementation measures chunk size using Python string length, so these values represent characters rather than tokenizer-based token counts.

## 6. Embedding Model – BGE Small

I chose `BAAI/bge-small-en-v1.5` through FastEmbed for generating document and query embeddings.

My reasons were:

- It is a relatively lightweight embedding model.
- It can run locally, so embedding does not require an external LLM API call.
- It keeps document ingestion independent of the OpenRouter generation provider.
- It provides a practical starting point for semantic retrieval.

I can evaluate other embedding models later if retrieval quality becomes a bottleneck.

## 7. Vector Database – ChromaDB

I selected ChromaDB as the vector store.

For this assignment, I wanted persistent local vector storage without introducing the operational overhead of managing a separate vector database service.

ChromaDB allows me to store embeddings along with chunk metadata and retrieve relevant chunks using semantic similarity.

I have kept the vector database behind a separate storage abstraction so that the implementation can be changed later if needed.

## 8. Hybrid Retrieval – Vector Search + BM25

I decided not to depend entirely on semantic vector search.

Semantic retrieval is useful when the wording of a question differs from the wording in the source document. However, exact keywords, technical terms, identifiers, and names can also be important.

To address both cases, I implemented hybrid retrieval:

- Vector search using embeddings stored in ChromaDB.
- Keyword-based retrieval using BM25.
- Reciprocal Rank Fusion to combine the two ranked result lists.

This gives the retriever two different ways to identify relevant document chunks.

## 9. Reciprocal Rank Fusion (RRF)

I chose Reciprocal Rank Fusion instead of directly combining BM25 scores and vector similarity scores.

The reason is that these retrieval methods produce scores on different scales, so directly adding their raw scores would require additional normalization.

RRF combines the rankings instead of comparing the original score values.

I started with an RRF constant of 60 and made it configurable.

This is another parameter I may revisit during retrieval evaluation.

## 10. Keeping the RAG Implementation Lightweight

I chose to implement the main ingestion, retrieval, and generation flow using plain Python rather than introducing a complete orchestration framework.

I wanted to understand how the individual components interact:

- Document loading
- Chunking
- Embedding generation
- Vector storage
- Keyword retrieval
- Hybrid ranking
- Context construction
- Prompt creation
- LLM response generation

This also makes it easier to inspect and debug individual stages.

I may consider a framework such as LangChain or LangGraph if the project grows to require more complex workflows, but I did not want framework abstractions to hide the fundamentals of RAG in this assignment.

## 11. Context Construction and Citations

I decided to attach a citation identifier to every retrieved chunk before passing it to the LLM.

For example, `[C1]`, `[C2]`, and so on.

The context also contains the source filename and page number where available.

The generation prompt instructs the model to cite factual claims using these identifiers and to avoid inventing sources or information.

I also introduced a context character limit to prevent an unnecessarily large context from being sent to the LLM.

I want to improve citation validation and context deduplication as part of the next iteration.

## 12. Frontend and Backend Separation

I chose FastAPI for the backend and Streamlit for the frontend.

FastAPI handles document upload, ingestion, retrieval, and question answering. Streamlit acts as a lightweight interface that communicates with the backend through HTTP APIs.

I deliberately kept the frontend separate so that another client, such as a React application, could use the same backend APIs in the future.

---

## Decisions I Want to Revisit

These are not necessarily problems with the current implementation. They are areas where I want to use evaluation rather than assumptions to make the next decision.

1. Compare different chunk sizes and overlap values.
2. Evaluate recursive chunking against semantic chunking.
3. Compare embedding models for retrieval quality.
4. Evaluate hybrid retrieval against vector-only and BM25-only retrieval.
5. Explore reranking retrieved chunks.
6. Improve citation validation and remove redundant context.
7. Evaluate retrieval quality using metrics such as Recall@K and MRR.
8. Evaluate answer faithfulness and relevance using a representative question-answer dataset.

The goal is to make these decisions based on observable results rather than choosing technologies simply because they are popular.
