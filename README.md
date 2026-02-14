# SGTradeClassificationRagBot

A small research / prototype repository implementing a retrieval-augmented generation (RAG) workflow for classifying Singapore trade-related documents. The project contains a lightweight RAG tool, a simple "naive" classification agent, prompt & model utilities, and a focus evaluation utilities to measure classifier performance.

This README provides a quick overview, installation and test instructions, usage examples, and notes for development.

## Contents / Project structure (important files)
- `src/sg_trade_ragbot/agents/naive_agent.py` — a simple classification agent used as a baseline.
- `src/sg_trade_ragbot/tools/RAGTool.py` — RAG tooling: document ingestion / retrieval + context assembly for LLM prompting.
- `src/sg_trade_ragbot/utils/prompts/prompts.py` — prompt templates used by agents and tools.
- `src/sg_trade_ragbot/utils/models/models.py` — model wrappers / helpers.
- `src/sg_trade_ragbot/utils/evals/` — evaluation configuration and evaluator utilities (includes `bare_config.yaml`).
- `tests/` — unit tests for agents, tools, and utils.

## Goals
- Provide a reproducible RAG pipeline for classifying trade text.
- Offer simple baseline agents and utilities to evaluate different retrieval / prompt strategies.
- Easy to iterate: change prompts, plug in different llms

## Requirements
- Python 3.13+

## Quick start (macOS)

### Evaluation (main entrypoint)
1. Clone the repository:
   - git clone <repo-url>
   - cd SGTradeClassificationRagBot

2. Create and activate a virtual environment:
   - python3 -m venv .venv
   - source .venv/bin/activate

3. Install the package in editable mode:
   - pip install -e .

   If you only want test requirements:
   - pip install -r requirements-dev.txt
   (or inspect `pyproject.toml` to install exact deps)

4. Run tests:
   - pytest -q

## Example usage snippets

- Using the RAG tool (conceptual example):
````python
# Example: assemble a retriever and query it
from sg_trade_ragbot.tools.RAGTool import RAGTool

# ...initialize RAGTool with your embedding/retrieval backend and LLM wrapper...
rag = RAGTool(index_path="path/to/index", llm_client=..., config=...)

# Ingest documents (if supported)
# rag.ingest(documents)

# Query
query = "Classify this trade record: <text here>"
response = rag.answer(query)
print(response)
````

- Using the naive agent (conceptual example):
````python
from sg_trade_ragbot.agents.naive_agent import NaiveAgent

agent = NaiveAgent()
text = "Import of electronics from Country X, HS code 85..."
label = agent.classify(text)
print("Predicted label:", label)
````

Note: the actual constructors / method names may differ slightly; see the implementation in `src/sg_trade_ragbot/...` for exact signatures. The repository includes unit tests which illustrate expected usage patterns (see `tests/`).

## Evaluation
- Configurations for evaluation lives in `src/sg_trade_ragbot/utils/evals/` (e.g. `bare_config.yaml`).
- Use `Evaluator` in `utils.evals.evaluator` to run benchmarks / collect metrics.

## Development notes
- Prompt templates are in `utils/prompts` — tweak these to change agent behavior.
- Models wrappers in `utils/models` abstract the LLM/embedding implementations. Swap in your preferred LLM client by implementing the required interface.
- Tests live in `tests/` and use pytest — run them frequently during development.

