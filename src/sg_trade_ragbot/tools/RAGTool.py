from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.retrievers import VectorIndexRetriever

from config import PROCESSED_DATA_DIR
from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput, RetrievalItem, RAGToolError, RetrievalValidationError

import threading

import logging

logger = logging.getLogger(__name__)

# Lightweight tool-call counter for tests / debugging, replace?
_tool_call_count = 0
_tool_call_lock = threading.Lock()


load_dotenv()


_INDEX = None


def _load_index():
    """
    Load the persisted index from disk. Ensure the processed directory exists.
    """
    global _INDEX

    if _INDEX is None:
        Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
        storage_context = StorageContext.from_defaults(persist_dir=str(PROCESSED_DATA_DIR))
        index = load_index_from_storage(storage_context)

        _INDEX = index

    return _INDEX


def _increment_tool_call_count() -> None:
    global _tool_call_count
    with _tool_call_lock:
        _tool_call_count += 1


def get_tool_call_count() -> int:
    """Return the number of times the agent-facing tool has been invoked."""
    with _tool_call_lock:
        return _tool_call_count


def reset_tool_call_count() -> None:
    """Reset the invocation counter to zero (useful between test runs)."""
    global _tool_call_count
    with _tool_call_lock:
        _tool_call_count = 0


# Chunking is an issue. I suspect that chunks are too large for the smaller models
def _rag_tool_helper(question: str, top_k: int = 3) -> RAGToolOutput:
    """
    Query the persisted LlamaIndex and return a JSON-encoded response string.

    Successful return value:
      - The RAGToolOutput pydantic model.

    Failure behavior:
      - On error the function raises a RAGToolError or RetrievalValidationError
    """
    _increment_tool_call_count()

    try:
        # for now limit 3 since chunks reach more tokens than accepted
        # in llama? But honestly it shouldn't really be reaching this
        if top_k > 3:
            top_k = 3

        index = _load_index()
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

        query_engine = RetrieverQueryEngine(retriever=retriever,
                                            response_synthesizer=response_synthesizer)

        response = query_engine.query(question)

        answer = str(response)

        retrievals = []

        for sn in getattr(response, "source_nodes", []) or []:
            node = getattr(sn, "node", sn)

            # Prefer get_content(), then get_text(), then node.text attribute, then str(node)
            text = None
            if hasattr(node, "get_content"):
                try:
                    text = node.get_content()
                except Exception:
                    text = None

            if not text and hasattr(node, "get_text"):
                try:
                    text = node.get_text()
                except Exception:
                    text = None

            if not text:
                text = getattr(node, "text", None) or str(node)

            node_id = (getattr(node, "id", None) or
                       getattr(node, "id_", None) or
                       getattr(node, "doc_id", None) or
                       "")
            try:
                item = RetrievalItem(id=str(node_id), text=str(text))
                retrievals.append(item)
            except Exception as e:
                # Build helpful debug message (truncate long text/repr)
                snippet = (str(text)[:200] + "...") if text and len(str(text)) > 200 else str(text)
                node_repr = repr(node)
                node_repr_snip = node_repr[:200] + "..." if len(node_repr) > 200 else node_repr
                msg = (
                    f"Failed to validate retrieval item for node_id={node_id!r}. "
                    f"Validation error: {e}. Text snippet: {snippet!r}. Node repr: {node_repr_snip!r}"
                )
                logger.exception(msg)
                raise RetrievalValidationError(msg) from e

        # Build the pydantic output model and return JSON
        output = RAGToolOutput(answer=answer, retrievals=retrievals)

        return output

    except Exception as e:
        raise RAGToolError(str(e)) from e


# @tool
def rag_tool(question: str, top_k: int = 5) -> str:
    """
    Query the persisted index for information and return a JSON-encoded response string.

    Successful return value:
      - The RAGToolOutput pydantic model.

    Failure behavior:
      - On error the function raises a RAGToolError
    """

    try:
        output = _rag_tool_helper(question, top_k=top_k)
        print("RAG Tool output: %s", output.model_dump_json())

        return output.model_dump_json()
    except RAGToolError as e:
        return f"RAG tool error: {e}"
