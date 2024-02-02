from __future__ import annotations

from typing import Any

from .models import estimate_text_tokens
from .types import ModelResponse, Usage


class AnthropicModel:
    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        *,
        input_cost_per_million: float = 3.0,
        output_cost_per_million: float = 15.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.client = client

    def _usage(self, input_tokens: int, output_tokens: int) -> Usage:
        cost = (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000
        return Usage(input_tokens, output_tokens, cost)

    def estimate_usage(self, prompt: str, max_output_tokens: int) -> Usage:
        return self._usage(estimate_text_tokens(prompt), max_output_tokens)

    def complete(self, prompt: str, *, max_tokens: int) -> ModelResponse:
        if self.client is None:
            from anthropic import Anthropic

            self.client = Anthropic()
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return ModelResponse(
            text, self._usage(message.usage.input_tokens, message.usage.output_tokens)
        )


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.client = client

    def _usage(self, input_tokens: int, output_tokens: int) -> Usage:
        cost = (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000
        return Usage(input_tokens, output_tokens, cost)

    def estimate_usage(self, prompt: str, max_output_tokens: int) -> Usage:
        return self._usage(estimate_text_tokens(prompt), max_output_tokens)

    def complete(self, prompt: str, *, max_tokens: int) -> ModelResponse:
        if self.client is None:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = completion.usage
        input_tokens = usage.prompt_tokens if usage else estimate_text_tokens(prompt)
        text = completion.choices[0].message.content or ""
        output_tokens = usage.completion_tokens if usage else estimate_text_tokens(text)
        return ModelResponse(text, self._usage(input_tokens, output_tokens))
