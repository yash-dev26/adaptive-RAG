from app.schemas.state import GraphState
from app.config.openaiConfig import openai_client

def single_query_rewrite_node(state: GraphState):
    print("[flow] entering single_query_rewrite_node")
    query = state.query

    SYSTEM_PROMPT = """You are a query rewriting assistant for a retrieval system. You will be given a user query that is ambiguous and could be interpreted in multiple ways. Your task is to rewrite the query to make it more specific and clear, while keeping the meaning as close as possible to the original. You must not answer the query, only rewrite it."""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],  
        temperature=0.5,
    )   

    rewritten_query = response.choices[0].message.content.strip()
    return {
        "rewritten_query": rewritten_query
    }
