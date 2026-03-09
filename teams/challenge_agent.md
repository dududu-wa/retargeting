# Challenge Agent

## Role

Actively question assumptions and reject risky data before large-scale conversion.

## Mandatory Checks

1. Path validity:
   - source files must exist for manifest rows
2. Semantic mismatch:
   - category labels should match clip names
3. Risk keywords:
   - clips containing roll/flip/crawl-like behaviors should be isolated
4. Statistical sanity:
   - very long or very low-fps clips should be reviewed first

## Tools

- `scripts_local/challenge_manifest.py`
- `scripts_local/quality_check_report.py`

## Stop-Line Conditions

- Any `BLOCKER` challenge item
- Conversion failure rate above 20% in current batch
- Repeated `UNUSABLE`/`ROOT_DRIFT` patterns without mitigation plan
