from __future__ import annotations

from langchain_openai import ChatOpenAI

from researchmind.config import (
    DEFAULT_MODEL,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    TEMPERATURE,
    get_dashscope_api_key,
)


def create_ali_chat_model(
    model_name: str | None = None,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model_name or DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        api_key=get_dashscope_api_key(),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
