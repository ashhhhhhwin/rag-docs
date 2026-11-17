from dataclasses import dataclass
from typing import Any

import httpx

from app.config import settings


class LLMProviderError(Exception):
    """Raised when the configured LLM provider fails."""


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any]


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or settings.openrouter_api_key
        self.model = model or settings.llm_model
        self.base_url = (base_url or settings.openrouter_base_url).rstrip("/")
        self.timeout_s = timeout_s if timeout_s is not None else settings.llm_timeout_s
        self.transport = transport

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured")

    async def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_s,
                transport=self.transport,
            ) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMProviderError(
                f"OpenRouter returned HTTP {exc.response.status_code}: {error_body}"
            ) from exc

        except httpx.RequestError as exc:
            raise LLMProviderError(f"Could not connect to OpenRouter: {exc}") from exc

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            model = data.get("model", self.model)
            usage = data.get("usage") or {}

        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError("OpenRouter returned an unexpected response format") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("OpenRouter returned an empty answer")

        return LLMResponse(
            content=content.strip(),
            model=model,
            usage=usage,
        )
