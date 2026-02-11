import pytest

from dotenv import load_dotenv

from src.agents.naive_agent import naive_agent
from src.tools.RAGTool import rag_tool
from llama_index.core.agent.workflow import FunctionAgent

load_dotenv()


@pytest.mark.asyncio
async def test_naive_agent_integration_call():
    # basic sanity checks
    assert naive_agent is not None
    assert isinstance(naive_agent, FunctionAgent)

    # Ensure the agent uses the project's RAG tool
    tools = getattr(naive_agent, "tools", [])
    assert (rag_tool in tools or any(getattr(t, "name", None) ==
            getattr(rag_tool, "name", None) for t in tools))

    prompt = "Please provide a short summary of the provided context and answer: What is 2+2?"

    # Prefer calling the agent directly, fall back to common methods inline.
    result = None

    if callable(naive_agent):
        try:
            result = naive_agent(prompt)
        except Exception:
            result = None

    if result is None and hasattr(naive_agent, "run"):
        result = await naive_agent.run(prompt)

    assert result is not None, "Agent did not return a response"

    text = result if isinstance(result, (str, bytes)) else str(result)
    assert text.strip() != "", "Agent returned an empty response"
