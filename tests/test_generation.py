
import httpx
import pytest

from app.generation.client import OpenRouterClient
from app.generation.service import GenerationService
from app.ingestion.models import Chunk
from app.retrieval.models import RetrievedChunk


@pytest.mark.asyncio
async def test_openrouter_client_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"

        body = __import__("json").loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"][0]["role"] == "system"

        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "content": "The document says this. [C1]"
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                },
            },
        )

    transport = httpx.MockTransport(handler)

    client = OpenRouterClient(
        api_key="test-key",
        model="test-model",
        base_url="https://openrouter.ai/api/v1",
        transport=transport,
    )

    result = await client.generate(
    [
        {"role": "system", "content": "You are a document assistant."},
        {"role": "user", "content": "Test question"},
    ])

    assert result.content == "The document says this. [C1]"
    assert result.model == "test-model"
    assert result.usage["prompt_tokens"] == 100


@pytest.mark.asyncio
async def test_generation_service_returns_citations():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "content": "The policy allows 20 days. [C1]"
                        }
                    }
                ],
                "usage": {},
            },
        )

    client = OpenRouterClient(
        api_key="test-key",
        model="test-model",
        base_url="https://openrouter.ai/api/v1",
        transport=httpx.MockTransport(handler),
    )

    service = GenerationService(client)

    chunk = Chunk(
        id="chunk-1",
        source="leave-policy.pdf",
        page=3,
        index=0,
        text="Employees are entitled to 20 days of annual leave.",
    )

    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.95,
        vector_score=0.88,
    )

    result = await service.answer(
        "How many days of annual leave are allowed?",
        [retrieved],
    )

    assert "[C1]" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].source == "leave-policy.pdf"
    assert result.sources[0].page == 3


@pytest.mark.asyncio
async def test_generation_service_handles_no_chunks():
    client = OpenRouterClient(
        api_key="test-key",
        model="test-model",
    )

    service = GenerationService(client)
    result = await service.answer("Any question?", [])

    assert "couldn't find enough information" in result.answer
    assert result.sources == []
