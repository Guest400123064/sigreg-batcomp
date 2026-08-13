# sh/ — The Scoring Suite (thin wrappers)

Scoring scripts for the review gate. Each script is a thin bash wrapper that invokes
the `tseprobe` CLI (or Python importing `tseprobe` — interface TBD). **All metric
implementations live in the public-facing `tseprobe` package** (versioned, tested,
PI-maintained); these scripts only wire evidence in and measurements out.

## Interface

Per script:

- **In:** evidence — paths, trackio `(project, run_id)` references, parameters.
- **Out:** measurement as JSON — number(s) + supporting artifact paths + diagnostics.

The suite is the codification of EPISTEME.md's deterministic clauses: the charter says
*what* must be measured; `tseprobe` pins *how*; these scripts are the executable entry
points the PI runs at review time. Script versions are pinned in claim records; suite
growth (new script, version, purpose) is journaled.

## Initial suite (per design doc §13)

| Script | Measures | Claim class |
|---|---|---|
| `decode_sufficiency.sh` | Linear-probe R² from embedding to target statistic, residual diagnostics, held-out evaluation | "statistic S is linearly decodable under conditions C" |
| `cluster_preservation.sh` | ARI and friends as functions of λ, rank, and temporal correlation ρ, across seeds | "regularization strength affects cluster structure in way W" |
| `temporal_consistency.sh` | Stability of representations for temporally adjacent states | supporting measurement |

*(no scripts implemented yet — skeleton only)*
