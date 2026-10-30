# Code changes and provenance

Scientific feature, fitting, prediction and metric equations are unchanged. The released entry point is the root `run.py`; copied component scripts retain unused historical CLI stages and are called through their existing functions.

| Released file | Original source | Change |
|---|---|---|
| `jetson/scripts/acquire.py` | `original_jetson/scripts/acquire.py` | Byte-for-byte unchanged. |
| `jetson/scripts/audit.py` | `original_jetson/scripts/audit.py` | Byte-for-byte unchanged. |
| `jetson/scripts/prepare_stage2.py` | `original_jetson/scripts/prepare_stage2.py` | Byte-for-byte unchanged. |
| `jetson/scripts/diagnose_stage2.py` | `original_jetson/scripts/diagnose_stage2.py` | Byte-for-byte unchanged. |
| `jetson/scripts/stage3_core.py` | `original_jetson/scripts/stage3_core.py` | Byte-for-byte unchanged. |
| `jetson/run.py` | `thermal_ml_final/run.py` | Byte-for-byte unchanged. |
| `intel/prepare.py` | `intel_12700_replication/run.py` | Byte-for-byte unchanged. |
| `intel/run.py` | `intel_physics_comparison/run.py` | IO paths use ignored local inputs/outputs and immutable model path; historical prediction checks optional; Parquet-byte gate replaced by raw-source hashes and numerical checks; save verification separately; fixed-refit predictions also checked against frozen replay. |
| `intel/test_run.py` | `intel_physics_comparison/test_run.py` | Paths adapted to the local input/output and immutable model locations; test equations unchanged. |

New glue: root `run.py` stages and hash-verifies user-supplied source files, redirects the original parsers’ IO globals, selects the existing 60-second horizon, builds prediction identifiers from generated features and saved registry keys (instead of requiring reference-prediction rows), and invokes unchanged predictors/metrics. It exposes only dataset, source directory and optional Intel refit arguments.

Root `test_run.py` adapts the original Stage 3 future-perturbation test and adds focused schema/count, paired-row, train-scaling and frozen-aggregate checks necessitated by the raw-source route. The original Intel three-test suite is retained with path changes only.

Artifacts: Jetson registry and reference tables are filtered to original held-out Ridge/neural experiments; retained parameter values are unchanged. Source manifests retain only needed identities, hashes, selection metadata and parser fields. Intel model values/reference metrics are unchanged; private preservation/environment bookkeeping and duplicated license metadata are omitted. No raw/derived observations are included.

No publisher implementation code was copied: the parser and learning scripts originated in this project’s research workflow. Publisher license text is retained separately. Full source/released code hashes are in `reference/code_provenance.json`.

Not changed: timestamp parsing or alignment, horizon tolerance, gap exclusions, temperature/power units, 11/8 feature definitions, original splits, equal-recording weighting, causal preprocessing, thermal feature forms, model selection/settings, saved coefficients, residual generation, bounds or scoring.

Not packaged: historical exploratory model families, complete Jetson nested fitting/search, optional supplementary raw pipelines, raw acquisition automation or report artwork containing measured traces. Supplementary aggregate outcomes remain included.
