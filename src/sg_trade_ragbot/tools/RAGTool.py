from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.retrievers import VectorIndexRetriever

from config import PROCESSED_DATA_DIR
from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput, RetrievalItem, RAGToolError


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


# Chunking is an issue. I suspect that chunks are too large for the smaller models
def _rag_tool_helper(question: str, top_k: int = 3) -> RAGToolOutput:
    """
    Query the persisted LlamaIndex and return a JSON-encoded response string.

    Successful return value:
      - The RAGToolOutput pydantic model.

    Failure behavior:
      - On error the function raises a RAGToolError
    """
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

        if hasattr(response, "source_nodes") and getattr(response, "source_nodes"):
            try:
                for sn in response.source_nodes:
                    node = getattr(sn, "node", sn)
                    # try node.get_text(), fallback to node.text or str(node)
                    try:
                        text = node.get_text()
                    except Exception:
                        text = getattr(node, "text", None) or str(node)
                    node_id = getattr(node, "id", None) or getattr(node, "doc_id", None) or ""
                    retrievals.append({"id": node_id, "text": text})
            except Exception:
                retrievals = []

        # Validate and convert to pydantic RetrievalItem objects
        validated_retrievals = []
        for r in retrievals:
            try:
                # Ensure id/text are strings (pydantic will validate types)
                rid = "" if r.get("id") is None else str(r.get("id"))
                rtext = "" if r.get("text") is None else str(r.get("text"))
                item = RetrievalItem(id=rid, text=rtext)
                validated_retrievals.append(item)
            except Exception:
                # Skip invalid retrieval entries
                continue

        # Build the pydantic output model and return JSON
        output = RAGToolOutput(answer=answer, retrievals=validated_retrievals)

        return output

    except Exception as e:
        raise RAGToolError(str(e)) from e


# @tool
def rag_tool(question: str, top_k: int = 5) -> str:
    """
    Agent-facing wrapper: returns JSON string on success or the legacy
    "RAG tool error: ..." string on failure.
    """
    try:
        output = _rag_tool_helper(question, top_k=top_k)

        return output.model_dump_json()
    except RAGToolError as e:
        return f"RAG tool error: {e}"
