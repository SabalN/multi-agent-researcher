"""
Wires the three agent nodes into a LangGraph pipeline and provides a CLI.

The graph is intentionally linear (Planner → Researcher → Synthesiser) — this
is the simplest topology that exercises a true multi-agent state machine. A
more advanced version could add a conditional edge ("more research needed?")
that loops back to the Researcher.

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from agents import planner_node, researcher_node, synthesiser_node
from state import ResearchState


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_graph():
    """Construct and compile the research pipeline graph."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("synthesiser", synthesiser_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "synthesiser")
    workflow.add_edge("synthesiser", END)

    return workflow.compile()


def run_research(question: str, save_output: bool = True) -> ResearchState:
    """Run the full pipeline for `question` and optionally write the report
    to `outputs/`."""
    graph = build_graph()

    print(f"\n{'='*70}")
    print(f"Research question: {question}")
    print('='*70 + "\n")

    initial_state: ResearchState = {"question": question, "log": []}
    t0 = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - t0
    final_state["elapsed_seconds"] = elapsed

    # Console summary
    print("\n--- Pipeline log ---")
    for line in final_state.get("log", []):
        print(f"  {line}")

    print(f"\n--- Sub-queries ({len(final_state.get('sub_queries', []))}) ---")
    for i, sq in enumerate(final_state.get("sub_queries", []), 1):
        print(f"  {i}. {sq['question']}")

    print(f"\n--- Sources retrieved: {len(final_state.get('search_results', []))} ---")
    print(f"--- Unique URLs cited: {len(final_state.get('citations', []))} ---")
    print(f"--- Elapsed: {elapsed:.2f}s ---")

    print("\n--- Final report ---\n")
    print(final_state.get("final_report", "(no report)"))

    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = "".join(c for c in question[:50]
                       if c.isalnum() or c in " -_").strip().replace(" ", "_")
        out_path = OUTPUT_DIR / f"{timestamp}_{slug}.md"
        out_path.write_text(
            f"# Research Report\n\n"
            f"**Question:** {question}\n"
            f"**Generated:** {datetime.now().isoformat()}\n"
            f"**Elapsed:** {elapsed:.2f}s\n\n"
            f"---\n\n{final_state.get('final_report', '')}\n"
        )
        print(f"\n[saved] {out_path}")

    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent research pipeline (Planner → Researcher → Synthesiser).")
    parser.add_argument("question", nargs="+",
                        help="The research question to investigate")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the report to outputs/")
    args = parser.parse_args()

    run_research(" ".join(args.question), save_output=not args.no_save)


if __name__ == "__main__":
    main()
