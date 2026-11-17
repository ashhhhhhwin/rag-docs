from dataclasses import dataclass
from typing import Any

from app.generation.client import OpenRouterClient
from app.generation.prompts import build_messages
from app.retrieval.models import RetrievedChunk


@dataclass(frozen=True)
class SourceReference:
    citation_id: str
    source: str
    page: int | None


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    sources: list[SourceReference]
    model: str
    usage: dict[str, Any]


class GenerationService:
    def __init__(
        self,
        client: OpenRouterClient,
        max_context_chars: int = 18000,
    ) -> None:
        self.client = client
        self.max_context_chars = max_context_chars

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[SourceReference]]:
        context_parts = []
        sources = []
        used_chars = 0

        for index, result in enumerate(chunks, start=1):
            chunk = result.chunk
            citation_id = f"C{index}"

            passage = (
                f"[{citation_id}] "
                f"Source: {chunk.source} | "
                f"Page: {chunk.page or 'N/A'}\n"
                f"{chunk.text}\n"
            )

            if used_chars + len(passage) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= 0:
                    break
                passage = passage[:remaining]

            context_parts.append(passage)
            used_chars += len(passage)

            sources.append(
                SourceReference(
                    citation_id=citation_id,
                    source=chunk.source,
                    page=chunk.page,
                )
            )

            if used_chars >= self.max_context_chars:
                break

        return "\n".join(context_parts), sources

    async def answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> GeneratedAnswer:
        if not chunks:
            return GeneratedAnswer(
                answer=(
                    "I couldn't find enough information in the provided documents to answer that."
                ),
                sources=[],
                model=self.client.model,
                usage={},
            )

        context, sources = self._build_context(chunks)
        messages = build_messages(question, context)

        response = await self.client.generate(messages)

        return GeneratedAnswer(
            answer=response.content,
            sources=sources,
            model=response.model,
            usage=response.usage,
        )
