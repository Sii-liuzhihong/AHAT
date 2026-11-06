"""Direct chat-completions client with bounded retries and no optional SDK."""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    success: bool
    content: str = ""
    error: str = ""
    retry_count: int = 0


class LLMClient(Protocol):
    def call(self, messages, model_name=None, api_key=None, **kwargs) -> LLMResponse: ...


class OpenAICompatibleClient:
    def __init__(self, base_url=None, api_key=None, model=None, timeout=120, max_retries=3,
                 temperature=0.0, max_tokens=4096, seed=None):
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("TGPO_LLM_MODEL", "")
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed if seed is not None else os.environ.get("TGPO_LLM_SEED")
        if self.max_retries < 1 or self.timeout <= 0:
            raise ValueError("max_retries and timeout must be positive")

    def call(self, messages, model_name=None, api_key=None, **kwargs):
        model = model_name or self.model
        if not model:
            return LLMResponse(False, error="Set TGPO_LLM_MODEL or provide model_name")
        payload = {"model": model, "messages": messages,
                   "temperature": kwargs.pop("temperature", self.temperature),
                   "max_tokens": kwargs.pop("max_tokens", self.max_tokens), **kwargs}
        if "seed" not in payload and self.seed is not None:
            payload["seed"] = int(self.seed)
        key = self.api_key if api_key is None else api_key
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        endpoint = self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"
        error_message = "Request failed"
        for attempt in range(self.max_retries):
            request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                content = result["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("chat response content must be a string")
                return LLMResponse(True, content, retry_count=attempt)
            except urllib.error.HTTPError as error:
                # Do not log headers or server response bodies, which may echo secrets.
                error_message = f"Chat endpoint returned HTTP {error.code}"
                if error.code < 500 and error.code not in (408, 429):
                    return LLMResponse(False, error=error_message, retry_count=attempt)
            except (urllib.error.URLError, TimeoutError, OSError):
                error_message = "Chat endpoint connection or timeout error"
            except (KeyError, IndexError, TypeError, ValueError):
                error_message = "Invalid chat-completions response"
            if attempt + 1 < self.max_retries:
                time.sleep(min(0.25 * 2 ** attempt, 2))
        return LLMResponse(False, error=error_message, retry_count=self.max_retries - 1)


def create_default_llm_client(use_cpu_server=True, **kwargs):
    """The old server selector is accepted; endpoint selection is now explicit."""
    return OpenAICompatibleClient(**kwargs)


OpenAIClient = OpenAICompatibleClient
