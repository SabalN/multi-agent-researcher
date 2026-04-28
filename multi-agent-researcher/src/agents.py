"""
The three agent nodes that make up the pipeline.

Each agent is a pure function: takes the current state, returns the parts of
the state it wants to update. LangGraph handles the merge.

  - Planner    → decomposes the question into 3-5 sub-queries
  - Researcher → executes web searches for every sub-query
  - Synthesiser → produces a structured Markdown report with citations

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import json

from state import ResearchState, SubQuery, SearchResult
from tools import llm_chat, llm_json, web_search


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a research planner. Given a user's research question, decompose it into 3-5 specific sub-questions that, when answered together, fully address the original question.

Rules:
1. Sub-questions must be answerable by a focused web search.
2. Sub-questions should cover different angles (e.g. context/background, current state, key examples or evidence, comparisons, future/implications).
3. Avoid duplication — every sub-question must add a distinct angle.
4. Keep each sub-question concise (max 20 words).

Output strictly as a JSON array. No prose before or after. Schema:
[
  {"question": "...", "rationale": "..."},
  ...
]
"""


def planner_node(state: ResearchState) -> dict:
    question = state["question"]
    user_msg = f"Research question: {question}\n\nDecompose into 3-5 sub-questions as JSON."

    sub_queries_raw = llm_json(PLANNER_SYSTEM, user_msg)
    if not isinstance(sub_queries_raw, list):
        raise ValueError("Planner did not return a list")

    sub_queries: list[SubQuery] = [
        {"question": item["question"], "rationale": item.get("rationale", "")}
        for item in sub_queries_raw
        if isinstance(item, dict) and "question" in item
    ][:5]  # cap at 5

    return {
        "sub_queries": sub_queries,
        "log": [f"Planner produced {len(sub_queries)} sub-queries"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Researcher
# ─────────────────────────────────────────────────────────────────────────────
def researcher_node(state: ResearchState) -> dict:
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {"search_results": [], "log": ["Researcher: no sub-queries to run"]}

    all_results: list[SearchResult] = []
    log_msgs = []

    for sq in sub_queries:
        q = sq["question"]
        try:
            results = web_search(q, max_results=4)
            for r in results:
                all_results.append({
                    "sub_query": q,
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:500],
                    "raw_content": r.get("content", ""),
                })
            log_msgs.append(f"Researcher: '{q[:60]}...' → {len(results)} results")
        except Exception as e:
            log_msgs.append(f"Researcher: '{q[:60]}...' FAILED — {e}")

    return {"search_results": all_results, "log": log_msgs}


# ─────────────────────────────────────────────────────────────────────────────
# Synthesiser
# ─────────────────────────────────────────────────────────────────────────────
SYNTHESISER_SYSTEM = """You are a research synthesiser. You will be given:
- the original research question
- a list of sub-questions
- web search results (title, URL, snippet) grouped under each sub-question

Produce a structured Markdown report with these sections:

# [Title — restate the question concisely]

## Executive Summary
3-5 sentences answering the original question directly.

## Key Findings
For each sub-question, a sub-section with the sub-question as the heading and 1-2 paragraphs synthesising the findings. Cite sources inline using numbered references like [1], [2], etc.

## Limitations
2-4 bullet points on what the available sources couldn't tell us, or where they disagreed.

## Sources
Numbered list of URLs in the order they were cited (matching the inline numbering).

Rules:
1. Use ONLY the supplied search results. No outside knowledge.
2. If results don't cover something, say so explicitly rather than fabricating.
3. Inline citations are mandatory for every factual claim.
4. Keep the report under 1000 words.
"""


def synthesiser_node(state: ResearchState) -> dict:
    question = state["question"]
    sub_queries = state.get("sub_queries", [])
    results = state.get("search_results", [])

    if not results:
        return {
            "final_report": (
                f"# Research Report — {question}\n\n"
                "**No search results were obtained.** The Researcher agent did not "
                "produce any sources, so a synthesis cannot be generated.\n"
            ),
            "citations": [],
            "log": ["Synthesiser: aborted — no search results"],
        }

    # Group results by sub-query for easier prompting
    grouped: dict[str, list[SearchResult]] = {}
    for r in results:
        grouped.setdefault(r["sub_query"], []).append(r)

    # Build the structured context
    parts = [f"Original question: {question}\n"]
    parts.append(f"Sub-questions ({len(sub_queries)}):")
    for i, sq in enumerate(sub_queries, 1):
        parts.append(f"  {i}. {sq['question']}")
    parts.append("\nSearch results:\n")
    for sq_text, items in grouped.items():
        parts.append(f"\n### Results for sub-query: {sq_text}\n")
        for r in items:
            parts.append(f"- TITLE: {r['title']}")
            parts.append(f"  URL: {r['url']}")
            parts.append(f"  SNIPPET: {r['snippet']}")
            parts.append("")

    user_msg = "\n".join(parts) + "\n\nProduce the structured Markdown report now."
    report = llm_chat(SYNTHESISER_SYSTEM, user_msg, max_tokens=2000)

    # Pull URL list from search results in order seen
    seen = []
    for r in results:
        if r["url"] and r["url"] not in seen:
            seen.append(r["url"])

    return {
        "final_report": report,
        "citations": seen,
        "log": [f"Synthesiser: produced {len(report.split())}-word report "
                f"with {len(seen)} unique sources"],
    }
