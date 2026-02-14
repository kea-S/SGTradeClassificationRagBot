# **SGTradeClassificationRagBot**

A research prototype implementing a retrieval-augmented generation (RAG) workflow for classifying Singapore trade-related documents. The project contains a lightweight RAG tool, a simple "naive" classification agent, prompt & model utilities, and focused evaluation utilities to measure classifier performance.

## **Key Features**

1. **Modular Architecture:** A production-ready Python codebase with clear separation of concerns (Agents, Tools, Parsers, Evaluation).  
2. **Containerized Workflow:** A fully working Dockerfile and docker-compose setup for reproducible testing.  
3. **Evaluation-First:** Integrated promptfoo configuration for rigorous testing of prompts against trade scenarios.  
4. **Auditor’s Log:** A trace of the agent’s Chain of Thought (CoT) for the test cases, showing how it handled ambiguity.

## **Project Structure**

* src/sg\_trade\_ragbot/agents/naive\_agent.py — A simple classification agent used as a baseline.  
* src/sg\_trade\_ragbot/tools/RAGTool.py — RAG tooling: document ingestion, retrieval, and context assembly.  
* src/sg\_trade\_ragbot/utils/prompts/prompts.py — Prompt templates used by agents and tools.  
* src/sg\_trade\_ragbot/utils/models/models.py — Model wrappers and helpers.  
* src/sg\_trade\_ragbot/utils/evals/ — Evaluation configuration (e.g., bare\_config.yaml) and utilities.  
* tests/ — Unit tests for agents, tools, and utils.

## **Goals**

* **Reproducible Pipeline:** Provide a containerized RAG pipeline for classifying trade text.  
* **Baseline Benchmarks:** Offer simple agents to evaluate different retrieval and prompting strategies.  
* **Iteration:** Make it easy to swap prompts, plug in different LLMs (OpenAI, Groq, etc.), and measure impact.

## **Requirements**

* **Docker Desktop** (Recommended)  
* **uv** (Optional, for local dependency management)  
* Python 3.13+ (If running locally without Docker)

## **Quick Start (Docker)**

This project is containerized to ensure consistent evaluations across different machines. It uses uv for dependency management and mounts configuration files so you can edit test cases without rebuilding the container.

### **1\. Setup Configuration**

The container requires API keys to function.

1. Copy the example environment file:  
   cp .env.example .env

2. Open .env and add your keys (e.g., OPENAI\_API\_KEY, GROQ\_API\_KEY).**Note:** Do not add file paths to .env. The container handles paths automatically.

### **2\. Running an Evaluation**

To run the promptfoo evaluation against the default configuration:  
docker compose up \--build

This will:

1. Build the image (installing all dependencies from uv.lock).  
2. Run the evaluation script.  
3. Print the results to your terminal.

### **3\. The "Live Edit" Workflow**

You do **not** need to rebuild the container to modify prompts or test cases.

1. Open src/sg\_trade\_ragbot/utils/evals/eval\_configs/bare\_config.yaml in your local editor.  
2. Modify your prompts, test cases, or variables.  
3. Save the file.  
4. Run docker compose up again.  
   * *The container sees your changes immediately via Docker volumes.*

### **4\. Managing Dependencies**

If you add a new library (e.g., spacy), you must rebuild the container for Docker to see it:  
\# 1\. Update lockfile locally  
uv add spacy

\# 2\. Rebuild container  
docker compose up \--build

## **Known Issues & Roadmap**

### **The Auditor’s Log (Chain of Thought)**

The system generates a trace of the agent’s Chain of Thought (CoT) to show how it handles ambiguity in trade documents.

* **Current Status:** These traces are currently visible in the promptfoo debug logs/container output.  
* **Todo:** Implement a structured export or cleaner visualization for the Auditor's Log in the final report.

### **Local Ollama Support**

* **Current Status:** The Docker configuration currently relies on external APIs (OpenAI, Groq). Local Ollama instances running on the host machine are not yet bridgeable to the container network in this release.  
* **Todo:** Add a dedicated Ollama service to docker-compose.yml for fully offline, local model evaluation.

## **Development Notes**

* **Prompts:** Tweak templates in utils/prompts to change agent behavior.  
* **Models:** Wrappers in utils/models abstract the LLM/embedding implementations. You can swap in your preferred LLM client by implementing the required interface.  
* **Testing:** Tests live in tests/ and use pytest. Run them frequently during development.