from typing import Annotated, Optional

from langgraph.graph import add_messages
from pydantic import BaseModel, Field

class GraphState(BaseModel):
    user_id: str
    query: str
    intent: Optional[str] = None
    file_id: Optional[str] = None
    response: Optional[str] = None
    messages: Annotated[list, add_messages] = Field(default_factory=list)
