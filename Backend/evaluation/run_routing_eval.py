"""
Eval: asserts on *which path through the graph* each query takes
(nodes visited, task_type/intent/eval_action reached), not on answer text
quality.

Prerequisite: `python -m evaluation.seed_fixtures` has been run at least
once (writes evaluation/fixtures/file_ids.json).

Usage (from Backend/):
    python -m evaluation.run_routing_eval

Results are written ONLY to evaluation/results/routing_eval_<UTC ts>.json
— nowhere else. A summary is also printed to stdout for convenience, but
that's just console output, not a second copy of the results.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

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


def _check_case(case: dict, trace: list[str], final: dict) -> list[str]:
    """Returns a list of failure reasons; empty list means the case passed."""
    failures = []
    expect = case["expect"]

    if "task_type" in expect and final.get("task_type") != expect["task_type"]:
        failures.append(f"task_type: expected {expect['task_type']!r}, got {final.get('task_type')!r}")

    if "intent" in expect and final.get("intent") != expect["intent"]:
        failures.append(f"intent: expected {expect['intent']!r}, got {final.get('intent')!r}")

    for node in expect.get("must_visit", []):
        if node not in trace:
            failures.append(f"must_visit: {node!r} was never visited (trace={trace})")

    for node in expect.get("must_not_visit", []):
        if node in trace:
            failures.append(f"must_not_visit: {node!r} was visited (trace={trace})")

    if "eval_action_in" in expect and final.get("eval_action") not in expect["eval_action_in"]:
        failures.append(
            f"eval_action: expected one of {expect['eval_action_in']}, got {final.get('eval_action')!r}"
        )

    if expect.get("fallback_branch") == "dynamic_on_tavily_configured":
        tavily_configured = bool(final.get("tavily_configured"))
        expected_branch = "web_search" if tavily_configured else "generate"
        if expected_branch not in trace:
            failures.append(
                f"fallback_branch: tavily_configured={tavily_configured} so expected "
                f"{expected_branch!r} in trace, got trace={trace}"
            )

    return failures


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

    trace: list[str] = []
    final: dict = {}

    async for chunk in graph.astream(state, config=invoke_config, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node_name, output in chunk.items():
            trace.append(node_name)
            if isinstance(output, dict):
                final.update(output)

    failures = _check_case(case, trace, final)

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "query": case["query"],
        "passed": len(failures) == 0,
        "failures": failures,
        "trace": trace,
        "observed": {
            "task_type": final.get("task_type"),
            "intent": final.get("intent"),
            "eval_action": final.get("eval_action"),
            "rewrite_attempts": final.get("rewrite_attempts"),
            "tavily_configured": final.get("tavily_configured"),
            "confidence": final.get("confidence"),
        },
    }


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in the environment to run this eval.")

    file_ids = _load_file_ids()
    cases = json.loads((EVAL_DIR / "cases" / "routing_cases.json").read_text())

    graph = build_graph(checkpointer=None)

    results = []
    for case in cases:
        print(f"[routing-eval] running: {case['id']}...")
        result = await _run_case(graph, case, file_ids)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  -> {status}  trace={result['trace']}")
        for reason in result["failures"]:
            print(f"     - {reason}")

    passed = sum(1 for r in results if r["passed"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"routing_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[routing-eval] {passed}/{len(results)} passed. Full results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
