"""
Eval: answer/retrieval *quality*, not routing. Runs each case in
cases/golden_set.json through the real graph and scores the result with
Ragas:
  - context_precision / context_recall — is retrieval finding the right
    chunks (and not too much irrelevant noise)?
  - faithfulness — does the generated answer only say things the
    retrieved context actually supports?
  - answer_relevancy — does the answer address what was actually asked?


Prerequisite: `python -m evaluation.seed_fixtures` has been run at least
once. Ragas' default judge LLM/embeddings need OPENAI_API_KEY in the
environment (same key, used two ways: once by the graph itself via
build_run_config, once by Ragas' own judge calls — these are separate
API calls billed separately, worth knowing since this eval is
meaningfully more expensive per case than the routing eval in
run_routing_eval.py).

Usage (from Backend/):
    pip install -r evaluation/requirements-eval.txt
    python -m evaluation.run_ragas_eval

Results are written ONLY to:
    evaluation/results/ragas_eval_<UTC ts>.csv
    evaluation/results/ragas_eval_<UTC ts>_summary.json
— nowhere else. Per-case scores are also printed to stdout, but that's
just console output, not a second copy of the results.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.agent.graph.graphBuilder import build_graph
from app.schemas.state import GraphState
from app.service.graphRunner import build_run_config

EVAL_DIR = Path(__file__).parent
FIXTURES_DIR = EVAL_DIR / "fixtures"
RESULTS_DIR = EVAL_DIR / "results"

EVAL_USER_ID = "eval-fixtures-user"  # must match seed_fixtures.py


def _load_file_ids() -> dict:
    path = FIXTURES_DIR / "file_ids.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python -m evaluation.seed_fixtures` first."
        )
    return json.loads(path.read_text())


async def _run_case(graph, case: dict, file_ids: dict) -> dict:
    file_ref = case.get("file_ref")
    file_id = file_ids[file_ref] if file_ref else None

    state = GraphState(user_id=EVAL_USER_ID, query=case["query"], file_id=file_id)

    api_keys = {
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "groq_api_key": os.environ.get("GROQ_API_KEY"),
    }
    thread_id = f"eval:{case['id']}:{datetime.now(timezone.utc).timestamp()}"
    invoke_config = build_run_config(thread_id, api_keys)
    invoke_config["configurable"]["token_queue"] = asyncio.Queue()  # generate_node expects one

    final = await graph.ainvoke(state, config=invoke_config)

    context_docs = final.get("context") or []
    contexts = [d["text"] for d in context_docs if isinstance(d, dict) and d.get("text")]
    if not contexts and final.get("summary_text"):
        # Summarize-task queries never populate `context` (summarize_node
        # condenses chunks internally and only returns summary_text) — the
        # condensed summary is what the final answer is actually grounded
        # on, per SUMMARY_SYSTEM_PROMPT in agent/prompts/generate.py, so
        # it's the right stand-in "retrieved context" for faithfulness
        # scoring here even though it isn't literally a retrieval result.
        contexts = [final["summary_text"]]

    return {
        "id": case["id"],
        "question": case["query"],
        "answer": final.get("response", ""),
        "contexts": contexts,
        "ground_truth": case["ground_truth"],
    }


async def _collect_samples(cases: list[dict], file_ids: dict) -> list[dict]:
    graph = build_graph(checkpointer=None)
    samples = []
    for case in cases:
        print(f"[ragas-eval] running graph for: {case['id']}...")
        sample = await _run_case(graph, case, file_ids)
        samples.append(sample)
    return samples


def _score_with_ragas(samples: list[dict]):
    # Imported lazily so `--help`-style usage and the rest of this file
    # stay importable even without evaluation/requirements-eval.txt
    # installed; the actual scoring step needs it.
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    dataset = Dataset.from_dict(
        {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
    )

    return evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment to run this eval.")

    file_ids = _load_file_ids()
    cases = json.loads((EVAL_DIR / "cases" / "golden_set.json").read_text())

    samples = asyncio.run(_collect_samples(cases, file_ids))

    print("[ragas-eval] scoring with Ragas (this makes additional judge-LLM calls)...")
    result = _score_with_ragas(samples)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    df = result.to_pandas()
    csv_path = RESULTS_DIR / f"ragas_eval_{ts}.csv"
    df.to_csv(csv_path, index=False)

    # Ragas' input-echo column names have shifted across versions —
    # older releases label them question/answer/contexts/ground_truth,
    # newer ones use user_input/response/retrieved_contexts/reference.
    # Rather than guess which naming this installed version uses, only
    # trust dtype: the four metrics we asked for (faithfulness,
    # answer_relevancy, context_precision, context_recall) are always
    # numeric, and everything else ragas echoes back is text. The
    # question text itself comes from `samples` (built locally, so its
    # keys are always what we defined below), matched back up by row
    # position — ragas preserves input row order in its output.
    metric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not metric_cols:
        raise RuntimeError(
            f"No numeric metric columns found in ragas output; got columns: {list(df.columns)}. "
            "Ragas' output schema may have changed — check evaluation/requirements-eval.txt's "
            "pinned version against what's actually installed."
        )

    per_case = []
    for sample, (_, row) in zip(samples, df.iterrows()):
        entry = {"id": sample["id"], "question": sample["question"]}
        entry.update({col: float(row[col]) for col in metric_cols})
        per_case.append(entry)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(samples),
        "mean_scores": {col: float(df[col].mean()) for col in metric_cols},
        "per_case": per_case,
    }
    summary_path = RESULTS_DIR / f"ragas_eval_{ts}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[ragas-eval] mean scores:")
    for metric, score in summary["mean_scores"].items():
        print(f"  {metric}: {score:.3f}")
    print(f"\n[ragas-eval] full results: {csv_path}")
    print(f"[ragas-eval] summary: {summary_path}")


if __name__ == "__main__":
    main()