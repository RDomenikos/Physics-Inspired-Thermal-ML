# Evaluation plan — frozen before any new model fits

Intel: fixed-configuration leave-one-date-out on the 36 originally valid recordings; the four rejected captures remain excluded. Jetson inference: ten original windows only, grouped conservatively by all audited raw-capture dates/overlap links. Groups are 22 October, 23 October and 24–28 October; five retained windows in the last group occur on 25 October. These are defensible dependence blocks, not proof of independent sessions. No observation-level random splitting. The manifest lists source ranges and overlap links.

## Exact outer folds

- intel / 2025-09-29: train R09, R10, R11, R12, R13, R14, R15, R22, R23, R24, R25, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39 (20484 origins); test R00, R01, R02, R04, R05, R06, R07, R08, R16, R17, R18, R20 (10105 origins).

- intel / 2025-09-30: train R00, R01, R02, R04, R05, R06, R07, R08, R09, R10, R11, R12, R13, R14, R15, R16, R17, R18, R20 (16140 origins); test R22, R23, R24, R25, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39 (14449 origins).

- intel / 2025-10-01: train R00, R01, R02, R04, R05, R06, R07, R08, R16, R17, R18, R20, R22, R23, R24, R25, R27, R28, R29, R30, R31, R32, R33, R34, R35, R36, R37, R38, R39 (24554 origins); test R09, R10, R11, R12, R13, R14, R15 (6035 origins).

- jetson_inference / G_2024-10-22: train phi_15W, phi_30W, phi_50W, phi_maxn, resnet_15W, resnet_maxn (1496 origins); test depth_15W, depth_30W, resnet_30W, resnet_50W (2675 origins).

- jetson_inference / G_2024-10-23: train depth_15W, depth_30W, phi_15W, phi_50W, phi_maxn, resnet_15W, resnet_30W, resnet_50W, resnet_maxn (4132 origins); test phi_30W (39 origins).

- jetson_inference / G_2024-10-24_to_2024-10-28: train depth_15W, depth_30W, phi_30W, resnet_30W, resnet_50W (2714 origins); test phi_15W, phi_50W, phi_maxn, resnet_15W, resnet_maxn (1457 origins).

## Procedures and bounded budget
Intel preserves fixed Ridge λ=10, correction λ=1/α=1; H2 width 2, λ=.03, correction λ=.1/α=.25, bound k=1. Five seeds 0–4. Each outer-training set contains two date groups. Generate residuals by fitting each baseline on one training group and predicting the other. Refit/freeze the receiver on both training groups; fit correction/scaling to those group-excluded residuals; derive bound from the training-input weighted q95 of the complete correction. Outer targets never participate. No selection/acceptance or tuning is added.

Jetson's original nested selection inside residual generation is infeasible with two outer-training groups: excluding a residual group leaves only one, and another inner validation split leaves zero training groups. User decision pending: either explicitly authorize fixed previously selected settings (same values as above, original Jetson solvers), or record nested corrected comparisons infeasible. No Jetson fits will run before resolving this choice.

Per feasible fixed-configuration dataset, cache at most 30 H2 baseline fits (three pairs and three single groups × five seeds), six Ridge baseline fits, and 18 correction fits. Preserve original optimizers (initial 2000, total allowance 8000); no extra restarts, seeds or searches. Store unsuccessful fits and mark affected fold/seed comparisons invalid, never silently remove them or replace OOF residuals with in-sample targets. For primary neural summaries require all five seeds valid in a fold.

## Scoring and scope
Within each test fold, use equal-recording MAE, sqrt(equal-recording MSE), >1°C error frequency and existing weighted inverse-ECDF tail metrics. Score each neural seed separately; average five seed scores for the fold. Across folds report arithmetic means and sample SD (ddof=1) of those fold metrics, with valid/planned fold counts. Mean fold RMSE is not pooled OOF RMSE. Do not pool hardware datasets, scopes or seeds. Paired improvement = baseline minus correction; percentages use that fold's baseline. Report all fold points and mean/sample SD of paired differences plus improved/tied/worsened counts (1e-10 numerical tie tolerance). Fold SD is descriptive with three dependence groups, not a confidence interval.

Reuse and verify the existing supplementary fifteen-window Jetson fixed-configuration grouped results, grouped separately under its original three blocks. No new supplementary fits. Retain its broader training-workload scope and adverse outcomes. Earlier-date test folds trained on later dates assess grouped generalization, not prospective forecasting. All data have informed prior development. Original report, white paper, code, results and publication package are protected; new local-only observations/predictions never enter the public companion ZIP.

## Authorized Jetson amendment — 28 October 2026, before Jetson fits

The earlier pending choice is resolved by explicit user authorization. Run **three-fold grouped cross-validation with previously selected, fixed hyperparameters**: Ridge λ=10, correction λ=1 and α=1; H2 width 2, λ=.03, correction λ=.1 and α=.25; bound k=1; seeds 0–4. All three dependence groups remain intact. For each outer fold, refit the baseline/preprocessing on the two training groups, generate residuals by leaving out each training group using fixed settings, fit the correction on those OOF residuals and derive the bound only from training inputs. No globally fitted coefficients, in-sample residuals, retuning or outer-test-driven changes. This is retrospective robustness assessment, not fully nested model-selection validation. The exact original nested procedure remains infeasible; the authorized fixed-configuration evaluation is distinct. Intel and supplementary model states are unchanged.
