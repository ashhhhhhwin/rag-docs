from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.dependencies import (
    get_generation_service,
    get_retriever,
    get_store,
)
from app.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentResponse,
    DocumentUploadResponse,
    SourceResponse,
)
from app.config import get_settings
from app.generation.client import LLMProviderError
from app.generation.service import GenerationService
from app.ingestion.loaders import SUPPORTED
from app.ingestion.pipeline import ingest_file
from app.retrieval.embeddings import get_embedder
from app.retrieval.retriever import Retriever
from app.retrieval.store import ChromaStore

router = APIRouter(prefix="/api/v1", tags=["RAG API"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile,
    store: ChromaStore = Depends(get_store),
):
    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if Path(filename).suffix.lower() not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT and Markdown files are supported.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File size cannot exceed 10 MB.",
        )

    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    destination = upload_dir / filename
    destination.write_bytes(content)

    try:
        chunks_created = await run_in_threadpool(
            ingest_file,
            destination,
            store,
            get_embedder(),
            settings.chunk_size,
            settings.chunk_overlap,
        )
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        ) from exc

    if chunks_created == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="No extractable text was found in the document.",
        )

    return DocumentUploadResponse(
        filename=filename,
        chunks_created=chunks_created,
        message="Document uploaded and indexed successfully.",
    )


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(
    store: ChromaStore = Depends(get_store),
):
    return [DocumentResponse(filename=filename) for filename in store.sources()]


@router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    generation_service: GenerationService = Depends(get_generation_service),
):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        retrieved_chunks = retriever.search(
            question,
            top_k=get_settings().top_k,
        )

        result = await generation_service.answer(
            question=question,
            chunks=retrieved_chunks,
        )

        return AskResponse(
            question=question,
            answer=result.answer,
            sources=[
                SourceResponse(
                    citation_id=source.citation_id,
                    source=source.source,
                    page=source.page,
                )
                for source in result.sources
            ],
            model=result.model,
            usage=result.usage,
        )

    except LLMProviderError as exc:
        # Detailed errors belong in server logs, not public API responses.
        print(f"LLM PROVIDER ERROR: {exc}")
        raise HTTPException(
            status_code=502,
            detail="The language model provider could not generate an answer.",
        ) from exc

    except Exception as exc:
        print(f"ASK ENDPOINT ERROR: {exc}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering the question.",
        ) from exc
