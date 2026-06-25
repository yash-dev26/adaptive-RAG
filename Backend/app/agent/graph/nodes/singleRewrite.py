from app.service.LLMProviders import generate_completion
from app.schemas.state import GraphState
from app.config.models import REWRITE_MODEL, REWRITE_PROVIDER

def single_query_rewrite_node(state: GraphState):
    print("[flow] entering single_query_rewrite_node")
    query = state.query
    history = list(state.messages or [])
    role_map = {"human": "User", "ai": "Assistant", "system": "System"}
    
    history_lines = []
    for msg in history[-6:]:
        role = role_map.get(getattr(msg, "type", ""), "User")
        content = getattr(msg, "content", "")
        if content:
            history_lines.append(f"{role}: {content}")
    
    history_block = "\n".join(history_lines) if history_lines else "No prior conversation."

    SYSTEM_PROMPT = f"""You are a query rewriting assistant for a retrieval system. You will be given a conversation history followed by the user's latest query. The latest query may contain ambiguous references like "it", "this", "they" that refer to something in the history. Rewrite the latest query into a fully self-contained, specific question that can be understood without the conversation history. Do not answer the query. Return only the rewritten query as plain text.
    Conversation History:
    {history_block}"""

    response_text = generate_completion(
        provider=REWRITE_PROVIDER,
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.5,
    )

    rewritten_query = response_text.strip()
    return {
        "rewritten_query": rewritten_query,
        "rewrite_type": "single",
    }
