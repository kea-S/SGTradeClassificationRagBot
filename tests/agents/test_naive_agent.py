import pytest

from dotenv import load_dotenv

from sg_trade_ragbot.agents.naive_agent import get_naive_agent
from sg_trade_ragbot.utils.models.models import REMOTE_LLAMA3, REMOTE_QWEN
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

    prompt = "Please provide the HS code for Live sheep, pure-bred and breeding"

    # Run the agent and expect a structured AgentOutput per LlamaIndex docs
    reset_tool_call_count()

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
    assert hasattr(response, "structured_response"), "Agent response missing structured_response"
    structured = response.structured_response
    assert structured is not None, "structured_response is empty"

    # Convert to the expected Pydantic model using the agent helper if available
    validated = None
    if hasattr(response, "get_pydantic_model"):
        validated = response.get_pydantic_model(RAGToolOutput)
    else:
        # Fallback: validate manually with pydantic
        if hasattr(RAGToolOutput, "model_validate"):
            validated = RAGToolOutput.model_validate(structured)
        else:
            validated = RAGToolOutput.parse_obj(structured)

    # Basic structural checks on the validated model
    assert isinstance(validated.answer, str) and validated.answer.strip() != ""
    assert isinstance(validated.retrievals, list)
    for item in validated.retrievals:
        assert hasattr(item, "id") and isinstance(item.id, str)
        assert hasattr(item, "text") and isinstance(item.text, str)
