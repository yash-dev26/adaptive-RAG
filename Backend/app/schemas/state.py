from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from typing_extensions import Annotated


class GraphState(BaseModel):
    user_id: str
    query: str
    file_id: Optional[str] = None

    expaneded_query: Optional[str] = None
    rewritten_query: Optional[str] = None

    intent: Optional[Literal["rag", "llm"]] = None

    context: Optional[List[str]] = None

    messages: Annotated[List, add_messages] = Field(default_factory=list)

    response: Optional[str] = None

    needs_retrieval: bool = False
    rewrite_type: Optional[Literal["none", "single", "multi"]] = None