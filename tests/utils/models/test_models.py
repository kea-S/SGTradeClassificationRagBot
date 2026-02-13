import pytest

from sg_trade_ragbot.utils.models.models import LANGCHAIN, LLAMAINDEX
from sg_trade_ragbot.utils.models.models import (
    REMOTE_QWEN,
    REMOTE_LLAMA3,
    REMOTE_OPENAI,
    LOCAL_LLAMA3,
)

from sg_trade_ragbot.utils.models.models import get_remote_llm, get_local_llm

from dotenv import load_dotenv


load_dotenv()


@pytest.mark.parametrize("model_name, local", [
    (REMOTE_LLAMA3, False),
    (REMOTE_QWEN, False),
    (REMOTE_OPENAI, False),
    (LOCAL_LLAMA3, True),
])
def test_llamaindex_llm_call(model_name, local):
    llm = (get_local_llm(model_name, LLAMAINDEX)
           if local else get_remote_llm(model_name, LLAMAINDEX))

    try:
        response = llm.complete("Hello")
        assert response is not None and len(str(response)) > 0
    except Exception as e:
        pytest.fail(f"LLAMAINDEX model '{model_name}' call failed: {e}")


@pytest.mark.parametrize("model_name, local", [
    (REMOTE_LLAMA3, False),
    (REMOTE_QWEN, False),
    (REMOTE_OPENAI, False),
    (LOCAL_LLAMA3, True),
])
def test_langchain_llm_call(model_name, local: bool):
    llm = (get_local_llm(model_name, LANGCHAIN)
           if local else get_remote_llm(model_name, LANGCHAIN))

    try:
        response = llm.invoke("Hello").text
        assert response is not None and len(str(response)) > 0
    except Exception as e:
        pytest.fail(f"LANGCHAIN model '{model_name}' call failed: {e}")
