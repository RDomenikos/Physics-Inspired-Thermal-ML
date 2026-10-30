# Verification summary

Both headline preparation/replay routes passed in an isolated copy using separate, hash-verified copies of the original publisher recordings. No original prepared feature table or historical row-level prediction file was supplied. Results were compared with frozen aggregate references from the completed research, not references generated during packaging.

| Check | Jetson | Intel |
|---|---:|---:|
| Prepared origins | 4,171 | 30,589 |
| Held-out origins | 1,457 | 6,035 |
| Frozen metric rows reproduced | 1,992 | 1,080 |
| Maximum MAE/RMSE difference | < 1e-20 °C | 0 °C |
| Focused/existing tests passed | 4 | 3 |

Scientific comparison tolerance: 1e-10 °C. Saved coefficients, scaling, split definitions, model settings and reference results were not retuned or overwritten. Source file identities were verified before parsing. Test coverage includes causal future-perturbation invariance, development-only scaling, timestamp/count constraints, complete paired rows, frozen references, baseline-plus-correction reconstruction and independent metrics. Intel tests also check training-group exclusion and training-derived bounds.

Intel fixed-procedure refitting was separately verified from the regenerated development features. It reproduced frozen prediction values exactly in this environment (maximum difference 0 °C); all aggregate reference checks passed. Refitted parameters were not promoted or saved over the released models. Jetson nested refitting and supplementary raw workflows are not packaged or claimed as tested.

Tested commands, from the isolated package root (source directories contained publisher recordings only):

```sh
python -B run.py jetson --source-dir ../source_recordings/jetson
python -B run.py intel --source-dir ../source_recordings/intel
python -B run.py jetson
python -B run.py intel --refit-check
python -B -m unittest test_run.py
python -B -m unittest discover -s intel -p test_run.py
```

The original-source download commands were not run: local copies of the exact pinned publisher bytes were used. Publisher metadata/license retrieval was checked online separately. Figshare archive extraction/equivalence is unverified; the verified Intel route uses the pinned GitHub byte identities. The environment used Python 3.9.12 and the pinned dependencies in `requirements.txt`, with one numerical thread.

`reference/verification.json` contains full reproduced headline values and separate replay/refit status. `CODE_CHANGES.md` documents every copied script and portability change. The final publication folder and ZIP are inspected directly, not merely filtered by `.gitignore`. No third-party recordings, derived observations, row-level predictions, measured traces, private filesystem paths or credentials are included. Original research folders and the earlier publication attempts remain unchanged.
