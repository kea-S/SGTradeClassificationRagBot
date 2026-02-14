import json
from dotenv import load_dotenv
from pathlib import Path
import importlib
import pytest
import config

load_dotenv()

# Skip the whole module if the processed index directory isn't present or empty
PROCESSED = Path(config.PROCESSED_DATA_DIR)
pytestmark = pytest.mark.skipif(
    not PROCESSED.exists() or not PROCESSED.is_dir() or not any(PROCESSED.iterdir()),
    reason=f"Processed index directory {PROCESSED!s} not present or empty; skipping integration test.",
)


def _call_tool_unwrapped(func, *args, **kwargs):
    """Call through decorator wrappers if present (e.g. @tool)."""
    if hasattr(func, "__wrapped__"):
        return func.__wrapped__(*args, **kwargs)
    if hasattr(func, "func"):
        return func.func(*args, **kwargs)
    return func(*args, **kwargs)


def test_rag_tool_with_real_index():
    """
    Integration test using the actual persisted LlamaIndex located at
    config.PROCESSED_DATA_DIR.

    This imports the real RAGTool module (which loads the persisted index at import time)
    and calls the tool with a simple question. The tool now returns a JSON string
    with the shape: {"answer": <str>, "retrievals": <list>}. The test verifies the
    returned value is valid JSON, contains the expected keys, that the answer is a
    non-empty string and not an error message, and that at least one retrieval is present.

    """
    # Import the module inside the test so the collection-time skip can prevent import failures
    RAGTool = importlib.import_module("sg_trade_ragbot.tools.RAGTool")

    question = "Provide the title of the index documents."
    # choose a sensible default top_k
    json_str = _call_tool_unwrapped(RAGTool.rag_tool, question, 5)

    try:
        payload = json.loads(json_str)
    except Exception:
        pytest.fail(f"rag_tool did not return valid JSON: {json_str!r}")

    assert isinstance(payload, dict)
    assert "answer" in payload and "retrievals" in payload

    answer = payload["answer"]
    retrievals = payload["retrievals"]

    assert isinstance(answer, str)
    assert answer.strip() != "", "rag_tool returned empty answer"
    assert not answer.startswith("RAG tool error:"), f"rag_tool returned error: {answer!s}"

    assert isinstance(retrievals, list)
    assert len(retrievals) != 0
