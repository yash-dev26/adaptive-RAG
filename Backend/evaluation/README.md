# Evaluation

Two separate eval scripts, testing two separate things. Neither is part of
the running API or the Docker image — this whole folder is dev/CI tooling

| | tests | cost | needs |
|---|---|---|---|
| `run_routing_eval.py` | **routing/state correctness** — did the query take the graph path it should have (task_type, intent, which nodes ran, fallback branch)? | cheap — only the LLM calls the graph itself already makes | just the app's own deps |
| `run_ragas_eval.py` | **answer/retrieval quality** — is the answer faithful to what was retrieved, relevant, and is retrieval finding the right chunks? | more expensive — Ragas makes its own judge-LLM calls on top of the graph's | `requirements-eval.txt` (ragas + datasets), see below |

Results are written **only** to `evaluation/results/` — nothing lands
anywhere else in the repo. A summary also prints to stdout for
convenience while a run is happening, but that's just console output,
not a second copy.

## Setup

Both scripts run against your real Qdrant/Redis/Mongo (whatever the app
itself is configured to use — same `.env` / environment) and need
`OPENAI_API_KEY` set, same as running the app normally.

1. Seed the fixture document (only needs to be done once, or again after
   wiping your Qdrant collection):

   ```bash
   cd Backend
   python -m evaluation.seed_fixtures
   ```

   This ingests `evaluation/fixtures/acme_handbook.txt` — a small, made-up
   handbook with deliberately exact, checkable facts (see the file itself)
   — through the app's real ingestion pipeline (same chunking + hybrid
   embedding + Qdrant upsert code `/api/v1/ingest` uses), under a
   dedicated `user_id` (`eval-fixtures-user`) that won't collide with real
   user data. Writes `evaluation/fixtures/file_ids.json`, which both eval
   scripts read.

2. Run the routing eval (no extra deps needed):

   ```bash
   python -m evaluation.run_routing_eval
   ```

3. Run the Ragas eval (needs the extra deps, see below):

   ```bash
   pip install -r evaluation/requirements-eval.txt
   python -m evaluation.run_ragas_eval
   ```

## Is ragas a prod dependency? No 

So: `evaluation/requirements-eval.txt` is separate from
`Backend/requirements.txt` on purpose, and `evaluation/` is excluded from
the Docker build via `.dockerignore`. Install
`requirements-eval.txt` into whatever runs the eval (a local venv, a
separate CI job) — not into the same environment that serves the app, and
never into the Docker image.

## Adding cases

- **Routing cases** (`cases/routing_cases.json`): each case names a query,
  optionally a `file_ref` (a key from `fixtures/file_ids.json`, or `null`
  for no-file queries), and an `expect` block — some combination of
  `task_type`, `intent`, `must_visit`/`must_not_visit` node lists,
  `eval_action_in`, or the special `"fallback_branch":
  "dynamic_on_tavily_configured"` (used for the one case that legitimately
  has two valid outcomes depending on whether Tavily is configured in your
  environment — the runner checks the branch that was actually taken
  against the `tavily_configured` flag observed in that same run, rather
  than hardcoding one).

- **Golden-set cases** (`cases/golden_set.json`): each case is a
  `(query, file_ref, ground_truth)` triple. Keep `ground_truth` short and
  specific — Ragas' `context_recall` metric checks whether the retrieved
  context could support each claim in it, so a vague ground truth makes
  that metric less meaningful.

Both fixture files use `evaluation/fixtures/acme_handbook.txt` right now.
If you add a second fixture document, add its `.txt` to `fixtures/`,
re-run `seed_fixtures.py`, and reference its filename (minus `.txt`) as
`file_ref` in new cases.

## Keeping a visible result

`results/` is gitignored — every run writes a new timestamped file, and
committing all of them would just accumulate stale, regenerated noise in
history (same reason you don't commit build output generally).

The one exception is `results/latest/`