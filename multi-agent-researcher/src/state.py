"""
Shared state for the multi-agent research pipeline.

The state is the single source of truth that gets passed between agents.
Each agent reads what it needs and writes its outputs back into the state.

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

from typing import TypedDict, Annotated
from operator import add


class SubQuery(TypedDict):
    """A single research sub-question planned by the Planner agent."""
    question: str       # Natural-language sub-question
    rationale: str      # Why the Planner picked this sub-question


class SearchResult(TypedDict):
    """A single web search result returned for a sub-query."""
    sub_query: str      # Which sub-query produced this
    title: str
    url: str
    snippet: str        # Tavily's content excerpt
    raw_content: str    # Full content if available, else snippet


class ResearchState(TypedDict, total=False):
    """The graph's shared state. Mutated as each node runs.

    Note on `Annotated[..., add]`: LangGraph uses these reducers to merge
    list updates from successive nodes (rather than overwriting). For the
    fields we treat as accumulating logs, we use `add`; otherwise updates
    overwrite the field as expected.
    """
    # Inputs
    question: str                              # User's research question

    # Planner outputs
    sub_queries: list[SubQuery]                # Decomposed sub-questions

    # Researcher outputs
    search_results: list[SearchResult]         # All retrieved sources

    # Synthesiser outputs
    final_report: str                          # Markdown report
    citations: list[str]                       # Numbered URL list

    # Audit / observability
    log: Annotated[list[str], add]             # Step-by-step trace
    elapsed_seconds: float                     # Total wall-clock time
