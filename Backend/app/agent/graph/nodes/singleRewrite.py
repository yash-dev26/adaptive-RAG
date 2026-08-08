from app.service.LLMProviders import generate_completion
from app.schemas.state import GraphState
from app.config.models import REWRITE_MODEL, REWRITE_PROVIDER
from app.agent.graph.keys import extract_keys
from app.agent.prompts.singleRewrite import build_single_rewrite_prompt
from langchain_core.runnables import RunnableConfig

async def single_query_rewrite_node(state: GraphState, config: RunnableConfig):
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

    SYSTEM_PROMPT = build_single_rewrite_prompt(history_block)

    openai_api_key, groq_api_key = extract_keys(config)

    response_text = await generate_completion(
        provider=REWRITE_PROVIDER,
        model=REWRITE_MODEL,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
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
