# Scientific audit changes — 2026-10-09

## Behavior

Add acquisition-scale conversion of optical motion proxies and a biological-unit bootstrap of equal-weight unit means. Expose scale via the CLI and HeartTwin adapter.

## Scope and remaining evidence

Only genuinely pixel-valued optical traces are scaled; standardized ensemble traces are rejected. Output is an optical proxy, not validated myocardial strain or force. The independent unit must be chosen and labeled correctly by the caller.

## Implementation

- `myotrace/physical_units.py`
- `tests/test_physical_units.py`
- `myotrace/cli.py`
- `myotrace/hearttwin_adapter.py`
- `myotrace/pipeline.py`
- `myotrace/uncertainty.py`

## Verification

Regression tests accompany the changes. Repository test results are recorded in the audit completion report and draft pull request. Software regression checks do not establish numerical, biological, transport or clinical validity.
