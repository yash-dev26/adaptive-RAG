"""Prompts for app/agent/graph/nodes/summarize.py."""

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
