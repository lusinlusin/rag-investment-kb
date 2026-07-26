"""Provider-agnostic text completion for the RAG toy.

Pick the backend with RAG_PROVIDER in .env  (deepseek | openai | claude).
Start with deepseek — cheapest. complete() has one signature for all three,
so rag.py never changes when you switch providers.

Claude uses the anthropic SDK; OpenAI and DeepSeek both use the openai SDK
(DeepSeek is OpenAI-compatible — same client, different base_url).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

PROVIDER = os.getenv("RAG_PROVIDER", "deepseek").lower()

# Default model per provider; override any of these in .env.
_MODELS = {
    "deepseek": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "claude": os.getenv("CLAUDE_MODEL", "claude-opus-4-8"),
}

# OpenAI-compatible providers: (base_url, api-key env var).
_OPENAI_COMPATIBLE = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),  # None -> openai's default base_url
}


def _require_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise RuntimeError(f"{env_var} is not set — add it to {Path(__file__).parent / '.env'}")
    return key


def complete(system: str, user: str, max_tokens: int = 4096) -> str:
    """Return the model's answer as plain text, for the selected provider.

    max_tokens is generous because reasoning models (e.g. deepseek-v4-flash) spend
    part of this budget on hidden reasoning tokens; too small a cap can leave no room
    for the actual answer, which then comes back empty.
    """
    # Path 1: Claude via the anthropic SDK (system prompt is its own argument).
    if PROVIDER == "claude":
        import anthropic

        client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=_MODELS["claude"],
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Claude returns a list of content blocks; take the first text block.
        return next((b.text for b in msg.content if b.type == "text"), "")

    if PROVIDER not in _OPENAI_COMPATIBLE:
        raise ValueError(f"Unknown RAG_PROVIDER={PROVIDER!r} (use deepseek | openai | claude)")

    # Path 2: OpenAI and DeepSeek share the openai SDK — same client, only the
    # base_url differs (system prompt goes in as the first message instead).
    from openai import OpenAI

    base_url, key_var = _OPENAI_COMPATIBLE[PROVIDER]
    client = OpenAI(api_key=_require_key(key_var), base_url=base_url)
    resp = client.chat.completions.create(
        model=_MODELS[PROVIDER],
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def describe() -> str:
    """Human-readable 'provider · model' for logging."""
    return f"{PROVIDER} · {_MODELS.get(PROVIDER, '?')}"
