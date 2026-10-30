# Factual report-update material

Retrospective intercept-removal refinement, evaluated on all three existing grouped folds without refitting. Saved alpha=1; only the correction offset is subtracted at prediction time. Baselines, slopes, preprocessing and groups remain unchanged. Original correction results remain beside it.

Mean fold MAE decreases 15.77% and mean fold RMSE 12.59% relative to baseline. These are reductions of mean fold errors, not averages of fold percentages. Both metrics improve in two of three folds; late October remains slightly worse than baseline. The variant was introduced after inspecting the original grouped failure. It has not been independently confirmed or evaluated on Intel or neural models.

Original fixed-split evidence, inference-only grouped Jetson, Intel leave-one-date-out, and supplementary Jetson inference/training workloads are separate scopes. Never pool hardware errors. Neural seeds are scored separately, then averaged within each fold; SD across three folds uses ddof=1. Seed SD columns are separate. Positive improvement means baseline minus method error. Historical fixed tables use method-minus-comparator columns; their sign is opposite. Refinement frozen-reference change columns also use method minus reference. New comparison CSVs use positive improvement consistently.

All folds, seeds and adverse outcomes are retained. Supplementary grouped aggregates are reused, not newly refitted; their raw preparation route is not supplied. Original Jetson neural correction worsens fixed-split MAE despite improving RMSE. Generalization and physical-mechanism claims remain limited. No energy savings are measured.

CSV files preserve per-recording and seed detail; PDF/PNG figures show fixed comparisons, all grouped fold points with mean ± sample SD, refinement fold and recording comparisons, and paired changes. Wins/ties/losses are in paired-summary tables and fixed_comparisons.csv. Baseline ties are numerical at 1e-10°C. These are descriptive results on previously inspected recordings, not independent replicates or confidence intervals.
