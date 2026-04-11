import json

from app.schemas.state import GraphState
from app.config.openaiConfig import openai_client

def multi_query_rewrite_node(state: GraphState):
    query = state.query

    SYSTEM_PROMPT = """
    You are a query rewriting assistant for a retrieval system.

    Your task is to generate 3 alternative queries that improve document retrieval.

    Guidelines:
    - Keep queries semantically similar to the original
    - Make them more specific and clear
    - Focus on improving search relevance
    - Do NOT answer the query

    Return ONLY a JSON array of strings.

    Example:
    [
    "query 1",
    "query 2",
    "query 3"
    ]
    """

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.5,
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)

    return {
        "queries": data.get("queries", [query])
    }