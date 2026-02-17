import json
from dotenv import load_dotenv
from pathlib import Path
import importlib
import pytest
import config
from typing import List

from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput, RetrievalItem

load_dotenv()


def _import_module():
    # import inside helper so pytest collection-time imports don't execute index load
    return importlib.import_module("sg_trade_ragbot.tools.RAGTool")


def test_load_index_creates_dir_and_loads_index(tmp_path, monkeypatch):
    """
    Verify _load_index creates the processed directory if missing and uses
    StorageContext.from_defaults + load_index_from_storage to return the index.
    """
    RAGTool = _import_module()

    # Ensure cached index is cleared for the test
    monkeypatch.setattr(RAGTool, "_INDEX", None, raising=False)

    processed_dir = tmp_path / "processed"
    # Make the module point at our temp processed dir
    monkeypatch.setattr(RAGTool, "PROCESSED_DATA_DIR", str(processed_dir), raising=False)

    # Replace StorageContext.from_defaults with a stub that records the persist_dir
    recorded = {}

    def fake_from_defaults(persist_dir=None):
        recorded["persist_dir"] = persist_dir
        return object()
    monkeypatch.setattr(RAGTool, "StorageContext", type("SC", (), {"from_defaults": staticmethod(fake_from_defaults)}), raising=False)

    # Make load_index_from_storage return a sentinel index
    sentinel = object()
    monkeypatch.setattr(RAGTool, "load_index_from_storage", lambda storage: sentinel, raising=False)

    # Call the loader
    result = RAGTool._load_index()

    assert result is sentinel
    # directory should have been created by the loader
    assert processed_dir.exists() and processed_dir.is_dir()
    # the storage context should have been constructed pointing at our dir
    assert str(processed_dir) == recorded["persist_dir"]


def test_load_index_propagates_file_not_found(tmp_path, monkeypatch):
    """
    If load_index_from_storage raises FileNotFoundError, _load_index should
    propagate that error (caller can handle it).
    """
    RAGTool = _import_module()

    monkeypatch.setattr(RAGTool, "_INDEX", None, raising=False)

    processed_dir = tmp_path / "processed_no_index"
    monkeypatch.setattr(RAGTool, "PROCESSED_DATA_DIR", str(processed_dir), raising=False)

    # Stub StorageContext so it doesn't touch disk beyond directory creation
    monkeypatch.setattr(RAGTool, "StorageContext", type("SC", (), {"from_defaults": staticmethod(lambda persist_dir=None: object())}), raising=False)

    # Simulate missing index file error from llama-index loader
    def fake_load_index_from_storage(storage):
        raise FileNotFoundError("docstore.json not found")
    monkeypatch.setattr(RAGTool, "load_index_from_storage", fake_load_index_from_storage, raising=False)

    with pytest.raises(FileNotFoundError):
        RAGTool._load_index()


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
    payload = _call_tool_unwrapped(RAGTool._rag_tool_helper, question, 3)

    assert isinstance(payload, RAGToolOutput)
    assert hasattr(payload, "answer") and hasattr(payload, "retrievals")

    answer = payload.answer
    retrievals = payload.retrievals

    assert isinstance(answer, str)

    assert answer.strip() != "", "rag_tool returned empty answer"
    assert not answer.startswith("RAG tool error:"), f"rag_tool returned error: {answer!s}"

    assert isinstance(retrievals, List)

    for retrieval in retrievals:
        assert isinstance(retrieval, RetrievalItem)
