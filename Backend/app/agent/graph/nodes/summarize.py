"""Map-reduce document condensation for the "summarize" task type.

This node's only job is compiling a whole (potentially large) document down
to something small enough to hand to the centralized `generate` node as
`summary_text`.
"""

from app.schemas.state import GraphState
from app.retrieval.retrieval import fetch_all_chunks_for_file
from app.service.LLMProviders import generate_completion
from app.agent.graph.keys import extract_keys
from app.config.models import REWRITE_PROVIDER, REWRITE_MODEL
from langchain_core.runnables import RunnableConfig
import asyncio

# Rough chars-per-batch budget for the map step. Not a real tokenizer count
# just a conservative multiplier (~4 chars/token) 
MAP_BATCH_CHAR_BUDGET = 12_000

MAP_SYSTEM_PROMPT = """You are summarizing one section of a larger document.
Write a dense, factual summary of ONLY the section below 
RULES:
1. do not editorialize,
2. do not say "this section discusses", 
3. just state the content directly.
4. Preserve specific names, numbers, and terms verbatim;they may be needed to
answer follow-up questions later. 
5. Keep it proportionate to the input length don't over-compress."""

REDUCE_SYSTEM_PROMPT = """You are given a sequence of section summaries from a
single document, in original document order. Combine them into one coherent,
well-organized condensation of the WHOLE document. Merge redundant points
across sections, preserve the overall structure/flow, and keep specific
facts, names, and numbers intact.

This is an intermediate artifact, not the final answer shown to the user —
a later step will turn it into the user-facing response, so favor
completeness and density over polished prose.

NOTE: Do not mention that you were given section summaries,
write as if summarizing the document directly."""

SINGLE_PASS_SYSTEM_PROMPT = """Condense the following document. Be
comprehensive but concise, preserve specific facts/names/numbers, and organize
the summary to reflect the document's own structure.

This is an intermediate artifact, not the final answer shown to the user —
a later step will turn it into the user-facing response, so favor
completeness and density over polished prose."""


def _batch_chunks(chunks: list[dict], char_budget: int) -> list[str]:
    batches = []
    current, current_len = [], 0
    for c in chunks:
        text = c["text"]
        if current and current_len + len(text) > char_budget:
            batches.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(text)
        current_len += len(text)
    if current:
        batches.append("\n\n".join(current))
    return batches


async def summarize_node(state: GraphState, config: RunnableConfig) -> dict:
    print("[flow] entering summarize_node")

    openai_api_key, groq_api_key = extract_keys(config)

    chunks = await fetch_all_chunks_for_file(state.file_id, state.user_id)

    if not chunks:
        # No LLM call needed for this one — `generate_node` still turns it
        # into the streamed final response, same as every other summary,
        # so the user always gets one consistent code path/UX regardless of
        # why the summary is short.
        return {
            "summary_text": "I couldn't find any ingested content for this document to summarize.",
        }

    batches = _batch_chunks(chunks, MAP_BATCH_CHAR_BUDGET)
    print(f"[summarize] {len(chunks)} chunks -> {len(batches)} map batch(es)")

    if len(batches) == 1:
        # Small enough to condense directly, single pass.
        summary_text = await generate_completion(
            provider=REWRITE_PROVIDER,
            model=REWRITE_MODEL,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            messages=[
                {"role": "system", "content": SINGLE_PASS_SYSTEM_PROMPT},
                {"role": "user", "content": batches[0]},
            ],
            temperature=0,
        )

        return {
            "summary_text": summary_text,
        }

    # Map: condense each batch in parallel on the cheap/fast provider —
    # these are intermediate artifacts, not user-facing.
    async def _summarize_batch(batch_text: str) -> str:
        return await generate_completion(
            provider=REWRITE_PROVIDER,
            model=REWRITE_MODEL,
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            messages=[
                {"role": "system", "content": MAP_SYSTEM_PROMPT},
                {"role": "user", "content": batch_text},
            ],
            temperature=0,
        )

    batch_summaries = await asyncio.gather(*[_summarize_batch(b) for b in batches])

    reduce_input = "\n\n---\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(batch_summaries)
    )

    # Reduce: combine section condensations into a single artifact that
    # generate_node will turn into the final, streamed, user-facing answer.
    summary_text = await generate_completion(
        provider=REWRITE_PROVIDER,
        model=REWRITE_MODEL,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        messages=[
            {"role": "system", "content": REDUCE_SYSTEM_PROMPT},
            {"role": "user", "content": reduce_input},
        ],
        temperature=0,
    )

    return {
        "summary_text": summary_text,
    }