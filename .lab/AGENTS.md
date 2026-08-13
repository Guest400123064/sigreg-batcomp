# AGENTS.md — Lab Control Block

*The authored input every agent in this lab inherits. The drill-level `AGENTS.md`
(minted by `fork.sh`) composes this block with a specific task assignment.*

## Mission

*(to be authored: the lab's research mission in a few sentences — time-series embedding
probing for LeJEPA/SIGReg-style objectives, aimed at a systematic analysis of what the
embeddings contain)*

## Roles and Write Permissions

Enforcement is topological, not instructional:

- **PI (master agent, kernel mode)** — the only entity that writes `states/`
  (`tasks.jsonl`, `arxiv.jsonl`, `index.lance/`) and `shared/`. Also the only entity
  that modifies `src/tseprobe/` (models, losses, synthetic data generators).
- **Student (subagent, user mode)** — works only inside its own `drills/<uuid>/`.
  Reads shared resources live; stages deliverables in `drills/<uuid>/yield/`.
  Never writes shared state, never invokes review verbs.

## Budget

*(to be authored: master meters in task-count chunks with human continue-gates; each
drill carries a self-proposed conceptual stopping point plus a liberal wall-clock leash
against rabbit-holing)*

## Rules

1. The archive is append-only; corrections are new records with `supersedes` edges.
2. Nothing in `shared/` is a research result; every shared asset carries provenance
   and a recomputation recipe.
3. A claim is not truth until the PI freezes it; student yields are staging, not record.
4. EPISTEME.md defines proof; amendments are human-approved only.
5. Evidence pointers use `(trackio project, run_id)` plus duplicated salient scalars.
