# HDM05 -> G1 Multi-Agent Execution

This folder defines a practical multi-agent workflow for the local retargeting pipeline.

## Agent Order

1. `Implementation Agent`: build and maintain local tooling.
2. `Integration Agent`: execute conversion runs and collect outputs.
3. `Challenge Agent`: challenge assumptions, block risky clips, and trigger rework.

## Shared Rules

- Keep all generated files traceable through CSV logs.
- Never run full dataset blindly; use staged batches (A -> B -> C).
- Every conversion batch must produce:
  - `logs/batch_convert_*/summary.csv`
  - quality template CSV
  - challenge report CSV

## Handoff Contract

- Implementation -> Integration:
  - scripts are runnable with `--help`
  - argument defaults are documented
- Integration -> Challenge:
  - manifest used for conversion
  - conversion summary and failed rows
  - first-pass quality report
- Challenge -> Implementation/Integration:
  - blocker list
  - required manifest edits
  - next-batch gate decision
