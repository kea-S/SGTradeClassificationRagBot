from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


load_dotenv()

# framework used
LANGCHAIN = "langchain"
LLAMAINDEX = "llama_index"

# models to be experimented with
REMOTE_LLAMA3 = "llama-3.3-70b-versatile"
REMOTE_QWEN = "qwen/qwen3-32b"   # mixtral is deprecated
REMOTE_OPENAI = "gpt-4o"


def get_remote_llm(name: str, framework: str):
    if name == REMOTE_OPENAI and framework == LANGCHAIN:
        return ChatOpenAI(model=REMOTE_OPENAI)
    if name == REMOTE_OPENAI and framework == LLAMAINDEX:
        return OpenAI(REMOTE_OPENAI)
    if framework == LLAMAINDEX:
        return Groq(name)
    elif framework == LANGCHAIN:
        return ChatGroq(model=name)


def get_local_llm(name: str, framework: str):
    pass
