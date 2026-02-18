import pytest
from pydantic import ValidationError

from dotenv import load_dotenv

from sg_trade_ragbot.agents.naive_agent import get_naive_agent
from sg_trade_ragbot.utils.models.models import REMOTE_LLAMA3, REMOTE_QWEN, REMOTE_GPT_OSS_SMALL
from sg_trade_ragbot.tools.RAGTool import rag_tool, get_tool_call_count, reset_tool_call_count
from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput

from llama_index.core.agent.workflow import FunctionAgent

load_dotenv()


@pytest.mark.asyncio
async def test_naive_agent_integration_call():
    naive_agent = get_naive_agent(REMOTE_QWEN, False)

    # basic sanity checks
    assert naive_agent is not None
    assert isinstance(naive_agent, FunctionAgent)

    # Ensure the agent uses the project's RAG tool
    tools = getattr(naive_agent, "tools", [])
    assert (rag_tool in tools or any(getattr(t, "name", None) ==
            getattr(rag_tool, "name", None) for t in tools))

    prompt = "Give me the HS Code for solar panels"

    # Run the agent and expect a structured AgentOutput per LlamaIndex docs
    reset_tool_call_count()

    naive_agent.dict()

    response = None
    try:
        response = await naive_agent.run(prompt)
    except Exception as e:
        # Always print the number of RAG tool invocations even if the run fails
        print("RAG tool calls during run (on exception):", get_tool_call_count())
        raise
    finally:
        # If run completed or after exception, also print the count
        print("RAG tool calls (final):", get_tool_call_count())

    assert response is not None, "Agent did not return a response"

    print("RAG tool calls during run:", get_tool_call_count())

    assert response is not None, "Agent did not return a response"

    # Prefer the agent's structured_response and pydantic conversion helper
    print(response)

    # Convert to the expected Pydantic model using the agent helper if available
    try:
        validated = RAGToolOutput.model_validate_json(str(response))

        assert validated.retrievals is not None
        assert validated.answer is not None
    except ValidationError as e:
        print(e.errors())
        raise
