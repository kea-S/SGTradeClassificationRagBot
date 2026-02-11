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
def rag_tool(question: str, top_k: int = 5, include_sources: bool = False) -> str:
    """
    Query a vector index retriever and return a text answer, optionally including source excerpts.

    This function builds a VectorIndexRetriever using the global INDEX, runs the query through
    INDEX.as_query_engine(...).query(...), and returns the result as a string. When include_sources
    is True the function attempts to extract source text from the returned response using one of
    two strategies (chosen at runtime depending on the response object's shape):
      - call response.get_formatted_sources() if available, or
      - iterate response.source_nodes and extract text from each node (handles different node
        shapes and nodes that expose a get_text() method).

    Parameters:
      question (str): The natural-language question to ask the retrieval-augmented generator.
      top_k (int): Maximum number of similar index entries to retrieve (passed to VectorIndexRetriever).
      include_sources (bool): If True, append a "Sources:" section with extracted source excerpts
                              to the returned answer when available.

    Returns:
      str: The answer as a string. If include_sources is True and sources are found, the answer
           will have a "Sources:" section appended. On error the function returns a string
           beginning with "RAG tool error: " followed by the exception message.

    Example:
      rag_tool("Summarize the contract terms", top_k=3, include_sources=True)
    """

    try:
        retriever = VectorIndexRetriever(index=INDEX, similarity_top_k=top_k)

        response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

        query_engine = RetrieverQueryEngine(retriever=retriever,
                                            response_synthesizer=response_synthesizer)

        response = query_engine.query(question)

        answer = str(response)

        sources = []
        if include_sources:
            if hasattr(response, "get_formatted_sources"):
                try:
                    sources = response.get_formatted_sources()
                except Exception:
                    sources = []
            elif hasattr(response, "source_nodes"):
                try:
                    for sn in response.source_nodes:

                        # guard for different node structures across versions
                        node = getattr(sn, "node", sn)
                        text = getattr(node, "get_text", None)
                        if callable(text):
                            sources.append(text())
                        else:
                            sources.append(str(node))
                except Exception:
                    sources = []

        if include_sources and sources:
            answer = answer + "\n\nSources:\n" + "\n".join(map(str, sources))

        return answer

    except Exception as e:
        return f"RAG tool error: {e}"
