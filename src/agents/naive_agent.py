from llama_index.core.agent.workflow import FunctionAgent

from src.tools.RAGTool import rag_tool
from src.utils.models.models import get_remote_llm, get_local_llm, LLAMAINDEX
from src.utils.prompts.prompts import NAIVE_AGENT_PROMPT


def get_naive_agent(model_name: str, local: bool = True):
    if local:
        llm = get_local_llm(model_name, LLAMAINDEX)
    else:
        llm = get_remote_llm(model_name, LLAMAINDEX)

    naive_agent = FunctionAgent(
        tools=[rag_tool],
        llm=llm,
        system_prompt=NAIVE_AGENT_PROMPT,
    )

    return naive_agent
