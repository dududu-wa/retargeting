# Integration Agent

## Goal

Run conversion batches safely and produce reproducible artifacts.

## Inputs

- manifest CSV
- GMR root
- target output dir

## Outputs

- converted `.pkl` files
- conversion summary log
- optional rendered videos

## Execution Checklist

1. Run with `--limit` first on a tiny subset.
2. Verify summary logs and output paths.
3. Expand to Batch A only after subset passes.
4. Run quality report before any batch expansion.
