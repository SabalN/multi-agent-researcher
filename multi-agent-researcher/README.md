# Multi-Agent Research Assistant (LangGraph)

Three coordinated AI agents — **Planner**, **Researcher**, and **Synthesiser** — orchestrated as a LangGraph state machine. Ask a research question, get a structured Markdown report with cited sources.

Built for the **Complex tier** of the COS6031-D Applied AI Professional portfolio (Element 3).

## Author
Sabal Nemkul · BSc (Hons) Applied Artificial Intelligence, University of Bradford

## What it does

1. **Planner agent** decomposes the user's research question into 3-5 specific sub-questions, each with a rationale
2. **Researcher agent** runs a web search (via Tavily) for every sub-question and gathers source material
3. **Synthesiser agent** consumes everything and produces a structured Markdown report with inline citations

The pipeline is wired up as a directed graph in LangGraph: **START → Planner → Researcher → Synthesiser → END**. Each node is a pure function that reads the shared state and returns updates to it.

## Why agents, not a single prompt?

A single prompt that says *"research this and write a report"* puts everything on the LLM at once: question understanding, search strategy, evidence gathering, and writing. Splitting into specialised agents has three benefits:

1. **Each agent has a focused job** with its own tailored system prompt → measurably better outputs at each stage
2. **The state is inspectable** — you can see exactly what sub-queries were produced, what was retrieved, and what was thrown away
3. **It's extensible** — adding a fact-checker agent or a re-planning loop is a graph edit, not a prompt rewrite

This is the architectural pattern most AI-engineering job adverts describe under "agentic AI" / "LangGraph" / "agent orchestration", and the reason I picked it for the complex-tier project.

## Architecture

```
                          ┌──────────────────────────┐
                          │   Shared ResearchState   │
                          │  (TypedDict, mutable)    │
                          └──────────────────────────┘
                                       ▲
                                       │ reads/writes
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌────────────────┐           ┌────────────────┐           ┌────────────────┐
│   Planner      │   ───►    │   Researcher   │   ───►    │  Synthesiser   │
│   (Claude)     │           │  (Tavily web   │           │   (Claude)     │
│                │           │   search +     │           │                │
│ Decomposes the │           │   returns      │           │ Builds         │
│ question into  │           │   ranked       │           │ structured     │
│ 3-5 sub-queries│           │   sources)     │           │ Markdown       │
│                │           │                │           │ report with    │
│                │           │                │           │ citations      │
└────────────────┘           └────────────────┘           └────────────────┘
```

**File layout**

| File | Responsibility |
|---|---|
| `src/state.py` | `ResearchState` TypedDict — the data passed between agents |
| `src/tools.py` | Anthropic LLM client + Tavily web search wrappers |
| `src/agents.py` | The three agent node functions |
| `src/graph.py` | Graph construction, CLI runner |
| `src/app.py` | Streamlit UI |

## Setup

```bash
# Clone, then:
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
# Edit .env to add ANTHROPIC_API_KEY and TAVILY_API_KEY
```

**API keys you'll need:**

| Service | Where to get one | Pricing |
|---|---|---|
| Anthropic | https://console.anthropic.com | Pay-as-you-go; pennies per query with Haiku |
| Tavily | https://tavily.com | Free tier: 1000 searches/month |

## Run

**Streamlit UI (recommended):**

```bash
streamlit run src/app.py
```

Open `http://localhost:8501`.

**CLI:**

```bash
python src/graph.py "What is the impact of the EU AI Act on UK startups?"
```

The CLI saves a Markdown report to `outputs/{timestamp}_{slug}.md`.

## Example output

> _Add a screenshot of the Streamlit UI showing a generated report here once you've run it._

For each question, the system produces:

- A **structured Markdown report** with executive summary, per-sub-question findings (with citations), limitations, and source list
- A **pipeline trace** showing the planner's sub-queries, every retrieved source, and timing per step

## What I learned building this

**1. State design is the hard part of agent systems.** Once the `ResearchState` TypedDict was right, each agent became almost trivial. The wrong state shape would have meant agents passing data through awkward nested dicts.

**2. The LangGraph wiring is mostly bookkeeping.** The actual orchestration code is ~10 lines. The complexity sits in the agent prompts and the state contract.

**3. JSON-mode prompting needs defensive parsing.** I had to strip Markdown code fences and validate types because Claude (sensibly) sometimes wraps JSON in ```` ```json ```` blocks despite explicit instructions not to. The `llm_json()` helper in `tools.py` handles this.

**4. Connection to my Industrial AI Project (Element 2).** The four-stage detection pipeline I built for my FYP (motion → tracker → event → classifier) is conceptually identical to this multi-agent graph (Planner → Researcher → Synthesiser): break a hard problem into specialised stages that filter and transform data, with a shared structure flowing between them. The same architecture pattern at a different abstraction level.

**5. Tavily over DuckDuckGo or scraping.** I considered a free DuckDuckGo wrapper or building a simple scraper. Tavily's purpose-built-for-agents API saved roughly a day of plumbing — clean content extraction, no rate-limit dance, structured JSON out. The free tier (1000 searches/month) is plenty for portfolio use.

## Limitations and what I would change

- **Linear graph only.** The current pipeline is `Planner → Researcher → Synthesiser`. A real research agent should have a conditional edge ("are these sources sufficient?") that loops back to the Researcher with refined queries. Adding it is a graph edit but adds complexity in evaluation.
- **No deduplication.** If two sub-queries return the same source, both copies are passed to the Synthesiser. A simple hash-based dedup would help.
- **No source-quality filter.** Tavily returns whatever ranks highly for the query. A filter for domain authority (e.g. exclude content farms, prefer government/academic sources) would improve quality.
- **No structured output validation.** The Synthesiser produces free-form Markdown. Forcing it through a JSON schema and rendering would make the output more reliable but less natural.
- **No formal evaluation.** Like most agent systems built quickly, I'm relying on qualitative inspection. A real-world deployment would build a rubric (correctness, citation quality, coverage) and human-rate a sample.

These limitations are exactly what a v2 would address — and exactly the kind of honest reflection my Element 1 feedback (62%) flagged me as needing more of.

## Tech stack

- **Orchestration:** LangGraph (StateGraph + START/END)
- **LLM:** Anthropic Claude (default: Haiku 4.5; configurable)
- **Web search:** Tavily API
- **UI:** Streamlit
- **Language:** Python 3.10+

## License

MIT — see `LICENSE` file.
