"""LLM client abstraction for multiple providers"""
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import httpx


@dataclass
class LLMConfig:
    provider: str = "ollama"  # ollama, openai
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        pass


class OllamaClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        import asyncio
        return asyncio.run(self.generate_async(prompt, system_prompt, **kwargs))

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    }
                }
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_key = config.api_key
        self.model = config.model

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        import asyncio
        return asyncio.run(self.generate_async(prompt, system_prompt, **kwargs))

    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


def create_llm_client(config: LLMConfig) -> LLMClient:
    providers = {
        "ollama": OllamaClient,
        "openai": OpenAIClient,
    }
    cls = providers.get(config.provider.lower())
    if not cls:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
    return cls(config)