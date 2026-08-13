# states/

This folder deliberately but explicitly mixes **source-of-truth** and **derived state**:

| Entry | Kind | Written by |
|---|---|---|
| `tasks.jsonl` | Source of truth — the queue. Location-as-state: a task's position in the system *is* its state. | PI only, append-only |
| `arxiv.jsonl` | Source of truth — frozen claim records (supported / refuted / inconclusive, with evidence and provenance). | PI only, via review verbs, append-only |
| `index.lance/` | Derived — searchable projection over `arxiv.jsonl`. Rebuildable at any time via `shared/bin/rebuild`. | PI-maintained (index-on-write) |
| `track/` | Managed substrate — trackio experiment tracking (`TRACKIO_DIR`). Knows nothing about claims; claims point into it by `(project, run_id)`. | trackio |

Crash recovery is a directory scan: pending tasks are in `tasks.jsonl`, in-flight work
is in `../drills/`, staged claims are in drill `yield/` folders, frozen claims are in
`arxiv.jsonl`.
