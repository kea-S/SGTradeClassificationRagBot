from llama_index.core.agent.workflow import FunctionAgent

from sg_trade_ragbot.tools.RAGTool import rag_tool
from sg_trade_ragbot.utils.models.models import get_remote_llm, get_local_llm, LLAMAINDEX, LANGCHAIN
from sg_trade_ragbot.utils.prompts.prompts import NAIVE_AGENT_PROMPT
from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput


def get_naive_agent(model_name: str, local: bool = True):
    if local:
        llm = get_local_llm(model_name, LLAMAINDEX)
    else:
        llm = get_remote_llm(model_name, LLAMAINDEX)

    naive_agent = FunctionAgent(
        tools=[rag_tool],
        llm=llm,
        system_prompt=NAIVE_AGENT_PROMPT,
        output_cls=RAGToolOutput,
        verbose=True,
    )

    return naive_agent
