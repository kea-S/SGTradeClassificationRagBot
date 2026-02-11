from llama_index.core.agent.workflow import FunctionAgent

from src.tools.RAGTool import rag_tool
from src.utils.models.models import LLAMAINDEX, REMOTE_LLAMA3, get_remote_llm
from src.utils.prompts.prompts import NAIVE_AGENT_PROMPT

naive_agent = FunctionAgent(
    tools=[rag_tool],
    llm=get_remote_llm(REMOTE_LLAMA3, LLAMAINDEX),
    system_prompt=NAIVE_AGENT_PROMPT,
)
