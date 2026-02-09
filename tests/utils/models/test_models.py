import pytest

from src.utils.models.models import LANGCHAIN, LLAMAINDEX
from src.utils.models.models import REMOTE_QWEN, REMOTE_LLAMA3, REMOTE_OPENAI

from src.utils.models.models import get_remote_llm

from dotenv import load_dotenv


load_dotenv()


@pytest.mark.parametrize("model_name", [
    REMOTE_LLAMA3,
    REMOTE_QWEN,
    REMOTE_OPENAI,
])
def test_llamaindex_llm_call(model_name):
    llm = get_remote_llm(model_name, LLAMAINDEX)

    try:
        response = llm.complete("Hello")
        assert response is not None and len(str(response)) > 0
    except Exception as e:
        pytest.fail(f"LLAMAINDEX model '{model_name}' call failed: {e}")


@pytest.mark.parametrize("model_name", [
    REMOTE_LLAMA3,
    REMOTE_QWEN,
    REMOTE_OPENAI,
])
def test_langchain_llm_call(model_name):
    llm = get_remote_llm(model_name, LANGCHAIN)

    try:
        response = llm.invoke("Hello").text
        assert response is not None and len(str(response)) > 0
    except Exception as e:
        pytest.fail(f"LANGCHAIN model '{model_name}' call failed: {e}")
