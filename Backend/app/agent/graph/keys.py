"""
BYOK keys travel through LangGraph's `config["configurable"]`, not through
GraphState. GraphState is what MongoDBSaver checkpoints on every step (it's
how conversation memory across turns works), putting API keys in it would
mean writing a user's OpenAI/Groq key to Mongo in plaintext on every message.
`configurable` is per-invocation only and is never part of the checkpointed
state schema, so it's the right channel for request-scoped secrets like BYOK keys.
"""

from typing import Optional, Tuple
from langchain_core.runnables import RunnableConfig


def extract_keys(config: Optional[RunnableConfig]) -> Tuple[str, Optional[str]]:
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    openai_api_key = configurable.get("openai_api_key")
    groq_api_key = configurable.get("groq_api_key")

    if not openai_api_key:
        raise RuntimeError(
            "openai_api_key missing from graph run config. "
            "chatService must pass it via `configurable` on every graph.invoke/astream call."
        )

    return openai_api_key, groq_api_key
