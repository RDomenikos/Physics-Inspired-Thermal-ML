# Grouped evaluation findings

Three-fold grouped cross-validation with previously selected, fixed hyperparameters. This is retrospective robustness assessment, not fully nested model-selection validation. Every fold refits its models. Neural summaries average seeds within folds first, then folds equally. SD uses ddof=1. No pooled OOF metric is substituted.

| Scope / method | MAE mean ± SD (°C) | Mean fold RMSE ± SD (°C) | Valid/planned folds |
|---|---:|---:|---:|
| Jetson — ten inference windows / H2 | 0.269349 ± 0.046976 | 0.357800 ± 0.087357 | 3/3 |
| Jetson — ten inference windows / H2 + bounded physics | 0.283891 ± 0.060473 | 0.361724 ± 0.097327 | 3/3 |
| Jetson — ten inference windows / H2 + physics | 0.308702 ± 0.089638 | 0.391993 ± 0.122860 | 3/3 |
| Jetson — ten inference windows / Ridge | 0.391497 ± 0.037690 | 0.453806 ± 0.021906 | 3/3 |
| Jetson — ten inference windows / Ridge + physics | 0.373550 ± 0.149162 | 0.457705 ± 0.193723 | 3/3 |
| Intel i7-12700 — 36 valid recordings / H2 | 3.554648 ± 0.439270 | 4.762842 ± 0.690632 | 3/3 |
| Intel i7-12700 — 36 valid recordings / H2 + bounded physics | 3.554213 ± 0.431100 | 4.749376 ± 0.685761 | 3/3 |
| Intel i7-12700 — 36 valid recordings / H2 + physics | 3.550943 ± 0.427297 | 4.743524 ± 0.681034 | 3/3 |
| Intel i7-12700 — 36 valid recordings / Ridge | 4.122811 ± 0.634884 | 5.625935 ± 0.753218 | 3/3 |
| Intel i7-12700 — 36 valid recordings / Ridge + physics | 3.948181 ± 0.577410 | 5.323949 ± 0.674343 | 3/3 |
| Jetson supplementary — fifteen inference/training windows / H2 | 0.380114 ± 0.171089 | 0.493108 ± 0.251819 | 3/3 |
| Jetson supplementary — fifteen inference/training windows / H2 + bounded physics | 0.392719 ± 0.201151 | 0.506628 ± 0.290419 | 3/3 |
| Jetson supplementary — fifteen inference/training windows / H2 + physics | 0.487827 ± 0.365679 | 0.685235 ± 0.599617 | 3/3 |

Paired improvements use baseline minus corrected error. Positive is better; no favorable folds or seeds were selected.

| Scope / correction | MAE improvement mean ± SD (°C) | RMSE improvement mean ± SD (°C) | MAE wins / ties / losses | RMSE wins / ties / losses |
|---|---:|---:|---:|---:|
| Intel i7-12700 — 36 valid recordings / H2 + bounded physics | 0.000435 ± 0.012349 | 0.013466 ± 0.007602 | 2/0/1 | 3/0/0 |
| Intel i7-12700 — 36 valid recordings / H2 + physics | 0.003705 ± 0.016229 | 0.019318 ± 0.012962 | 2/0/1 | 3/0/0 |
| Intel i7-12700 — 36 valid recordings / Ridge + physics | 0.174630 ± 0.145759 | 0.301986 ± 0.159868 | 3/0/0 | 3/0/0 |
| Jetson — ten inference windows / H2 + bounded physics | -0.014542 ± 0.020231 | -0.003924 ± 0.010041 | 1/0/2 | 1/0/2 |
| Jetson — ten inference windows / H2 + physics | -0.039353 ± 0.042884 | -0.034193 ± 0.057004 | 1/0/2 | 1/0/2 |
| Jetson — ten inference windows / Ridge + physics | 0.017946 ± 0.184117 | -0.003899 ± 0.211954 | 2/0/1 | 2/0/1 |
| Jetson supplementary — fifteen inference/training windows / H2 + bounded physics | -0.012605 ± 0.030635 | -0.013520 ± 0.039284 | 1/0/2 | 2/0/1 |
| Jetson supplementary — fifteen inference/training windows / H2 + physics | -0.107713 ± 0.194785 | -0.192128 ± 0.347995 | 1/0/2 | 2/0/1 |

Percent improvements and their sample SD are in paired_summary.csv; each fold percentage has its own baseline denominator.

Jetson inference has only 39 origins in the 23 October group, versus 2,675 on 22 October and 1,457 in the linked late-October group. These are heterogeneous, small dependence blocks, not independent trials. Intel date folds also change workload composition. Earlier held-out dates trained on later dates are not prospective forecasts.

The supplementary fifteen-window Jetson results are reused and independently checked, with no new fits. Their scope includes training workloads and remains separate. The original fixed-split comparison figure uses different training/evaluation compositions; it is not an additional paired improvement estimate.

The original reports, models, results and publication folder are unchanged. Refer to failure records for any unresolved fits or infeasible procedures. No confidence intervals based on rows/seeds are claimed.

## Jetson inference: all three folds

H2 values are means across five separately scored seeds. All three folds are valid. The late-October fold trains on the original development recordings, but its corrections use whole-group OOF residuals rather than the original recording-window residual procedure; equal receiver scores do not imply an identical correction.

| Held-out group | Method | MAE (°C) | Balanced RMSE (°C) |
|---|---|---:|---:|
| G_2024-10-22 | H2 | 0.311190 | 0.384753 |
| G_2024-10-22 | H2 + bounded physics | 0.323163 | 0.393157 |
| G_2024-10-22 | H2 + physics | 0.392779 | 0.483859 |
| G_2024-10-22 | Ridge | 0.424734 | 0.476236 |
| G_2024-10-22 | Ridge + physics | 0.300046 | 0.356026 |
| G_2024-10-23 | H2 | 0.218534 | 0.260143 |
| G_2024-10-23 | H2 + bounded physics | 0.214252 | 0.252565 |
| G_2024-10-23 | H2 + physics | 0.214383 | 0.252439 |
| G_2024-10-23 | Ridge | 0.399210 | 0.452718 |
| G_2024-10-23 | Ridge + physics | 0.275405 | 0.335991 |
| G_2024-10-24_to_2024-10-28 | H2 | 0.278323 | 0.428505 |
| G_2024-10-24_to_2024-10-28 | H2 + bounded physics | 0.314258 | 0.439450 |
| G_2024-10-24_to_2024-10-28 | H2 + physics | 0.318944 | 0.439681 |
| G_2024-10-24_to_2024-10-28 | Ridge | 0.350547 | 0.432464 |
| G_2024-10-24_to_2024-10-28 | Ridge + physics | 0.545200 | 0.681098 |

| Held-out group | Correction | MAE improvement (°C / %) | RMSE improvement (°C / %) |
|---|---|---:|---:|
| G_2024-10-22 | Ridge + physics | 0.124688 / 29.357% | 0.120211 / 25.242% |
| G_2024-10-22 | H2 + physics | -0.081588 / -26.218% | -0.099107 / -25.759% |
| G_2024-10-22 | H2 + bounded physics | -0.011973 / -3.847% | -0.008405 / -2.184% |
| G_2024-10-23 | Ridge + physics | 0.123805 / 31.012% | 0.116727 / 25.784% |
| G_2024-10-23 | H2 + physics | 0.004151 / 1.899% | 0.007704 / 2.962% |
| G_2024-10-23 | H2 + bounded physics | 0.004282 / 1.959% | 0.007578 / 2.913% |
| G_2024-10-24_to_2024-10-28 | Ridge + physics | -0.194653 / -55.528% | -0.248634 / -57.492% |
| G_2024-10-24_to_2024-10-28 | H2 + physics | -0.040621 / -14.595% | -0.011176 / -2.608% |
| G_2024-10-24_to_2024-10-28 | H2 + bounded physics | -0.035935 / -12.911% | -0.010945 / -2.554% |

In inference-only Jetson CV, Ridge correction improves mean MAE slightly but worsens mean RMSE; it improves both metrics in two folds and worsens both in the late-October fold. Both H2 corrections worsen mean MAE and mean RMSE, improving both only in the 23 October fold (39 origins). Bounding reduces the average deterioration relative to unbounded H2 correction. These outcomes are preserved without retuning.
