"""
Shared tools for the agent pipeline.

Wraps the Anthropic and Tavily clients in tiny convenience functions so the
agent code stays focused on logic rather than SDK plumbing.

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import json
import os

from anthropic import Anthropic


# ─────────────────────────────────────────────────────────────────────────────
# LLM client (singleton-ish — reuse across agents)
# ─────────────────────────────────────────────────────────────────────────────
_client: Anthropic | None = None


def get_llm() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or use a .env file."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def llm_chat(system: str, user: str,
             model: str = "claude-haiku-4-5-20251001",
             max_tokens: int = 1500) -> str:
    """Single-turn chat completion. Used by all three agents."""
    client = get_llm()
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def llm_json(system: str, user: str,
             model: str = "claude-haiku-4-5-20251001",
             max_tokens: int = 1500) -> dict | list:
    """Helper for when we want strict JSON output. Strips markdown fences
    if the model returns them and falls back to a raw-string error if
    parsing fails so the caller can decide what to do."""
    raw = llm_chat(system, user, model, max_tokens).strip()
    # Strip ```json ... ``` fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Drop opening fence (and optional language tag) and closing fence
        raw = "\n".join(lines[1:-1]) if lines[-1].strip().startswith("```") \
            else "\n".join(lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON. Error: {e}\n"
            f"--- Raw output ---\n{raw}\n--- End ---"
        ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Web search via Tavily
# ─────────────────────────────────────────────────────────────────────────────
def web_search(query: str, max_results: int = 4) -> list[dict]:
    """Search the web with Tavily and return a list of result dicts.

    Tavily was chosen because it's purpose-built for AI agents — it returns
    cleaned content excerpts rather than raw HTML. The free tier (1000
    searches/month) is more than enough for portfolio use.

    Falls back to a clear error if TAVILY_API_KEY is not set.
    """
    try:
        from tavily import TavilyClient
    except ImportError as e:
        raise RuntimeError(
            "tavily-python not installed. Run: pip install tavily-python"
        ) from e

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY not set. Sign up at https://tavily.com (free tier "
            "available) and add the key to your .env file."
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
    )
    return response.get("results", [])
