import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from llama_index.core import StorageContext, load_index_from_storage, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.retrievers import VectorIndexRetriever

from config import PROCESSED_DATA_DIR


load_dotenv()


def load_tool():
    storage_context = StorageContext.from_defaults(persist_dir=str(PROCESSED_DATA_DIR))
    index = load_index_from_storage(storage_context)

    return index


INDEX = load_tool()


# @tool
def rag_tool(question: str, top_k: int = 5) -> str:
    """
    Query the persisted LlamaIndex and return a JSON-encoded response string.

    Successful return value:
      - A JSON string with the shape:
        {
          "answer": "<textual answer from the index/query engine>",
          "retrievals": [
            {"id": "<node id or doc id>", "text": "<source excerpt or full text>"},
            ...
          ]
        }

    Failure behavior:
      - On error the function returns a plain (non-JSON) string beginning with
        "RAG tool error: " followed by the exception message.

    Parameters:
      question (str): Natural-language question to ask the retrieval-augmented generator.
      top_k (int): Maximum number of similar index entries to retrieve (passed to VectorIndexRetriever).

    Returns:
      str: JSON-encoded payload on success, or an error string on failure.

    Example:
      rag_tool("Provide the title of the index documents.", top_k=5)
    """

    try:
        retriever = VectorIndexRetriever(index=INDEX, similarity_top_k=top_k)

        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

        query_engine = RetrieverQueryEngine(retriever=retriever,
                                            response_synthesizer=response_synthesizer)

        response = query_engine.query(question)

        answer = str(response)

        retrievals = []
        # primary path for 0.14.x: source_nodes -> node.get_text()
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

        # simple fallback: formatted sources (display-oriented)
        if not retrievals and hasattr(response, "get_formatted_sources"):
            try:
                formatted = response.get_formatted_sources()
                if isinstance(formatted, (list, tuple)):
                    for i, s in enumerate(formatted):
                        retrievals.append({"id": f"formatted-{i}", "text": str(s)})
                elif formatted:
                    retrievals.append({"id": "formatted", "text": str(formatted)})
            except Exception:
                pass

        # Always return JSON so callers can parse structured data.
        payload = {"answer": answer, "retrievals": retrievals}
        return json.dumps(payload)

    except Exception as e:
        return f"RAG tool error: {e}"
