from typing import List
from pydantic import BaseModel, Field


class RetrievalItem(BaseModel):
    id: str = Field(..., description="Node or document id for the retrieval")
    text: str = Field(..., description="Source excerpt or full text for the retrieval")


class RAGToolOutput(BaseModel):
    answer: str = Field(..., description="Textual answer produced by the index/query engine")
    retrievals: List[RetrievalItem] = Field(default_factory=list, description="List of retrieved source items")

    @classmethod
    def from_tool_response(cls, raw: str) -> "RAGToolOutput":
        if isinstance(raw, str) and raw.startswith("RAG tool error:"):
            raise ValueError(raw)

        return cls.model_validate_json(raw)
