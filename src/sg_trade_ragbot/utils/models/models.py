from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from dotenv import load_dotenv


load_dotenv()

# framework used
LANGCHAIN = "langchain"
LLAMAINDEX = "llama_index"

# models to be experimented with
REMOTE_LLAMA3 = "llama-3.3-70b-versatile"
REMOTE_QWEN = "qwen/qwen3-32b"   # mixtral is deprecated
REMOTE_OPENAI = "gpt-4o"

REMOTE_JUDGE = "gpt-5"

LOCAL_LLAMA3 = "llama3.1:latest"


def get_remote_llm(name: str, framework: str):
    if name == REMOTE_OPENAI and framework == LANGCHAIN:
        return ChatOpenAI(model=REMOTE_OPENAI)
    if name == REMOTE_OPENAI and framework == LLAMAINDEX:
        return OpenAI(REMOTE_OPENAI)
    if framework == LLAMAINDEX:
        return Groq(name)
    elif framework == LANGCHAIN:
        return ChatGroq(model=name)


# llama-3.1-8b is not "good enough" at the current state,
# might be that
# 1. Chunks are too big
# 2. Too many similarity searches
def get_local_llm(name: str, framework: str):
    if framework == LANGCHAIN:
        return ChatOllama(
            model=name
        )
    if framework == LLAMAINDEX:
        return Ollama(
            model=name,
            request_timeout=30
        )
