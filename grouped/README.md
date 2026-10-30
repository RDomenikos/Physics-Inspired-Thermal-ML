# Thermal ML grouped-validation companion

Additional retrospective grouped evaluation; the original fixed-split report, white paper and publication package remain unchanged. This companion refits within outer training groups rather than replaying globally fitted models on different test folds.

**Status:** all three Intel folds and all three inference-only Jetson folds are complete. The existing fifteen-window supplementary Jetson evaluation is reused without refitting. Jetson uses **“three-fold grouped cross-validation with previously selected, fixed hyperparameters,”** as explicitly authorized before its fits. This is retrospective robustness assessment, not fully nested model-selection validation. The original nested-procedure constraint remains documented in `results/jetson_infeasibility.json`; no weakening of groups or in-sample residual substitution was used. All fixed-configuration fits converged.

Read `results/findings.md`, `results/summary.csv` and `results/paired_summary.csv`. These keep hardware and workload scopes separate. `evaluation_plan.md`, `groups.csv` and `folds.json` document source grouping and exact memberships. Aggregate figures are in `figures/`.

## Integration and original data

Place this folder alongside the existing `thermal_ml_github` package. It supplements that repository; it does not replace any existing files. Reuse the original package's verified raw-data preparation launcher. No third-party recordings, prepared features, targets, row-level predictions or temperature traces are distributed here.

Source credits and links:

- **Jetson:** BroncoSpace Lab / SCALES contributors, https://github.com/BroncoSpace-Lab/scales-docs , pinned commit `f78280e6995e3c62488ea32f4dd7bfc5990370d8`; recordings under `docs/data/`. No explicit license file was found at that pinned source; public availability is not a redistribution grant.
- **Intel:** Ahmed Abd Elaziz Elsayed, Abdullah Azhar Al-Obaidi and Hany E. Z. Farag. Dataset https://doi.org/10.6084/m9.figshare.31654879.v1 ; article https://doi.org/10.1038/s41597-026-07496-6 ; authors' source https://github.com/Ahmed-Elsayed95/High-resolution-AI-Data-Center-Training-Workloads-Dataset , pinned commit `f7f8d62e806ce2b64e93c58fce2bb18604f067ad`.

Intel's Figshare deposit lists CC BY 4.0; the pinned GitHub repository has a separate CC BY-NC-ND 4.0 notice. These are not assumed interchangeable. The tested original preparation uses the pinned GitHub source bytes. Source notices and recorded checks are retained in `reference/`. Obtain originals directly under their publisher's terms; do not mirror them in this companion.

## Reproduce

Python 3.9–3.11; install `requirements.txt` into a local virtual environment. One numerical CPU thread is used. No new hyperparameter grid, neural width, seed, optimizer budget or physics feature is introduced.

Obtain the original recordings using the main package's README. Its preparation creates ignored local feature files. From the main package directory, for Intel:

```sh
python -B run.py intel --source-dir local_sources/intel-workloads
python -B run.py jetson --source-dir local_sources/scales-docs
```

Then from this companion directory:

```sh
python -B run.py --dataset intel --features ../thermal_ml_github/intel/local/features.parquet --output-dir local/results
python -B run.py --dataset jetson_inference --features ../thermal_ml_github/jetson/local/features.parquet --output-dir local/results
python -B present.py --results-dir local/results --figures-dir local/figures
```

These commands freshly refit three outer folds per dataset, using the original fixed Intel settings and previously selected, authorized fixed Jetson settings and group-excluded residual construction. Outputs go into ignored `local/`, leaving the delivered aggregate references unchanged. The source path may be changed to another locally prepared feature file from the same original pipeline. The launcher verifies recording membership/counts and assigns only the frozen manifest groups.

To recompute descriptive tables from an existing local run:

```sh
python -B run.py --summarize --output-dir local/results
```

Published supplementary seed-level aggregate scores are reused by that command. No supplementary observations or historical row-level predictions are required. The supplementary numerical reference was checked against the original saved OOF predictions during local assembly, not regenerated from its own summary table.

Jetson settings are Ridge λ=10 with correction λ=1/α=1; H2 λ=.03 with correction λ=.1/α=.25; bound k=1; seeds 0–4. Each outer training set contains two intact groups. The baseline and preprocessing are refitted on permitted training data; residual producers train on one group and predict the other; correction scales and bounds use training data only. No globally fitted coefficients are reused. Original nested correction tuning is infeasible with this group count, so this authorized fixed-configuration assessment does not validate the model-selection procedure. Settings were not revised after observing outcomes.

`python -B -m unittest test_run.py` checks the delivered memberships and numerical summaries. Full fitted-state, preprocessing and row-level checks run automatically on newly fitted local datasets; they need local observations. No dataset is downloaded automatically.

## Metrics and interpretation

Per-fold MAE and RMSE use equal-recording weights. Neural scores are computed per seed and averaged within each fold. Across folds the summary reports arithmetic means and **sample SD**, with valid/planned fold counts. Seed SD is recorded separately and is not treated as between-fold variation. No pooled OOF RMSE replaces mean fold RMSE.

Paired improvement is baseline error minus corrected error. Percentages use each fold's own baseline denominator; mean percentage improvement is not the percentage change of aggregate means. All wins, numerical ties and losses are retained. The 1e-10 tie tolerance only classifies numerical ties; it does not control model acceptance.

There are three conservative dependence blocks per scope. Intel date folds change workload composition. Jetson's isolated 23 October inference group has only 39 valid origins. Folds training on later dates to test earlier dates assess grouped generalization, not prospective forecasting. These recordings informed earlier development, so the new evaluation is retrospective. Fold SD is descriptive; no row/seed-based confidence interval is claimed.

The original supplementary Jetson study covers fifteen inference/training-workload windows, remains separate, and retains its negative results. Comparing the original fixed split with grouped means compares different train/test compositions and estimands, not two methods on identical evaluation rows.

## Code and publication scope

`source/` contains exact function bodies copied from the original project scripts, with only imports and existing configuration globals supplied. `code_provenance.json` records original file hashes and individual function-text hashes. `run.py` adds grouping orchestration, cache/provenance handling and summaries; `present.py` draws aggregate plots; `test_run.py` adds focused verification. No scientific function was rewritten.

MIT applies to original project code; third-party notices and data terms remain separate. Existing publication contents and reports are preserved. The companion ZIP is checked by inspecting its actual members, not relying solely on `.gitignore`. It includes aggregate/per-recording metrics and fitted coefficient states, not downloaded or generated row-level data. Nothing is published or pushed.
