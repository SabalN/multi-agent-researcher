"""
Streamlit UI for the multi-agent research pipeline.

Run with:
    streamlit run src/app.py

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph import run_research  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Researcher",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Multi-Agent Research Assistant")
st.caption(
    "Three agents — **Planner**, **Researcher**, **Synthesiser** — orchestrated "
    "with LangGraph. Ask a research question and the system will decompose it, "
    "search the web, and produce a cited Markdown report."
)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("API keys")
    st.success("ANTHROPIC_API_KEY ✓") if os.getenv("ANTHROPIC_API_KEY") \
        else st.error("ANTHROPIC_API_KEY missing")
    st.success("TAVILY_API_KEY ✓") if os.getenv("TAVILY_API_KEY") \
        else st.error("TAVILY_API_KEY missing")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        """
        1. **Planner** decomposes your question into 3-5 sub-queries
        2. **Researcher** runs a web search for each sub-query (Tavily)
        3. **Synthesiser** writes a structured, cited Markdown report

        Each agent is a node in a LangGraph state machine.
        """
    )

    st.divider()
    st.subheader("Sample questions")
    samples = [
        "What is the impact of AI on the UK job market in 2025?",
        "How are EU AI Act regulations affecting startups?",
        "What's the state of edge AI deployment in retail?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True, key=f"sample_{s[:20]}"):
            st.session_state.question = s


# ─────────────────────────────────────────────────────────────────────────────
# Main input
# ─────────────────────────────────────────────────────────────────────────────
question = st.text_area(
    "Research question",
    value=st.session_state.get("question", ""),
    height=80,
    placeholder="e.g. What are the main applications of large language models in healthcare?",
)

go = st.button("🚀 Run research", type="primary", disabled=not question.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Execute
# ─────────────────────────────────────────────────────────────────────────────
if go:
    if not os.getenv("ANTHROPIC_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        st.error("Both ANTHROPIC_API_KEY and TAVILY_API_KEY must be set in your .env.")
        st.stop()

    with st.spinner("Running the agent pipeline (≈30-60s)…"):
        try:
            result = run_research(question, save_output=False)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    # ── Top-line metrics ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sub-queries", len(result.get("sub_queries", [])))
    col2.metric("Search results", len(result.get("search_results", [])))
    col3.metric("Sources cited", len(result.get("citations", [])))
    col4.metric("Time (s)", f"{result.get('elapsed_seconds', 0):.1f}")

    # ── The report ──
    st.divider()
    st.subheader("📄 Final report")
    st.markdown(result.get("final_report", "(no report)"))

    # ── Pipeline trace ──
    with st.expander("🔍 Pipeline trace (planner → researcher → synthesiser)"):
        for line in result.get("log", []):
            st.code(line, language="text")

        st.markdown("**Sub-queries the Planner produced:**")
        for i, sq in enumerate(result.get("sub_queries", []), 1):
            st.markdown(f"{i}. **{sq['question']}**")
            if sq.get("rationale"):
                st.caption(f"_Rationale: {sq['rationale']}_")

        st.markdown("**All retrieved sources:**")
        for i, r in enumerate(result.get("search_results", []), 1):
            st.markdown(f"**[{i}] [{r['title']}]({r['url']})**")
            st.caption(f"_From sub-query: {r['sub_query']}_")
            st.caption(r["snippet"][:300] + ("…" if len(r["snippet"]) > 300 else ""))
            st.divider()

    # ── Download ──
    st.download_button(
        "💾 Download report as Markdown",
        data=result.get("final_report", ""),
        file_name=f"research_report.md",
        mime="text/markdown",
    )
