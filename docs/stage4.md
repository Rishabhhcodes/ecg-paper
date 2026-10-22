# Stage 4 — The single evaluation on DS2

Runs the four locked configurations on DS2, the held-out set of 22 patients. The
purpose is to report generalisation to unseen patients after model choices were
made using DS1. This evaluation completed once; **do not rerun it**.

## Run and integrity checks

- Dataset verification: **21 passed, 0 failed, 0 warnings**.
- DS1: 50,977 beats across 22 patients.
- DS2: 49,668 beats across 22 patients.
- DS1 and DS2 share no patients.
- Evaluation script commit reported by the run: `1d69436ec1`.
- Completed output: `stage4_results.json` and `RECEIPT.json` in `artifacts/stage4/`.
- The run took approximately 9.2 minutes.
- Calibration transfer was not reported because the Stage 2 prediction files were absent.

## DS2 class counts

| Class | Beats |
|---|---:|
| N | 44,218 |
| S | 1,836 |
| V | 3,219 |
| F | 388 |
| Q | 7 |

## Configurations and results

Macro-F1 is reported as mean ± standard deviation across seeds where multiple runs
were used. `Se / +P` means sensitivity / positive predictivity.

| Configuration | Runs | Macro-F1 | N Se / +P | S Se / +P | V Se / +P | F Se / +P |
|---|---:|---:|---:|---:|---:|---:|
| CNN + class weighting + RR | 5 | 0.6181 ± 0.0113 | 0.885 / 0.991 | 0.877 / 0.386 | 0.900 / 0.881 | 0.428 / 0.081 |
| CNN + focal loss + RR | 5 | 0.5837 ± 0.0400 | 0.775 / 0.995 | 0.872 / 0.368 | 0.904 / 0.842 | 0.701 / 0.049 |
| XGBoost + class weighting + RR | 1 | 0.6194 | 0.960 / 0.969 | 0.371 / 0.391 | 0.924 / 0.924 | 0.351 / 0.148 |
| Logistic regression, no correction + RR | 1 | 0.5531 | 0.958 / 0.955 | 0.344 / 0.567 | 0.719 / 0.736 | 0.186 / 0.069 |

## Per-seed CNN results

### CNN + class weighting + RR

| Seed | Macro-F1 | S sensitivity | S positive predictivity | V sensitivity | V positive predictivity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.6068 | 0.819 | 0.414 | 0.885 | 0.705 |
| 1 | 0.6151 | 0.938 | 0.373 | 0.912 | 0.935 |
| 2 | 0.6283 | 0.899 | 0.330 | 0.919 | 0.870 |
| 3 | 0.6108 | 0.915 | 0.318 | 0.848 | 0.964 |
| 4 | 0.6295 | 0.814 | 0.497 | 0.934 | 0.930 |

### CNN + focal loss + RR

| Seed | Macro-F1 | S sensitivity | S positive predictivity | V sensitivity | V positive predictivity |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5445 | 0.905 | 0.454 | 0.854 | 0.788 |
| 1 | 0.6126 | 0.862 | 0.374 | 0.911 | 0.878 |
| 2 | 0.5488 | 0.815 | 0.276 | 0.916 | 0.735 |
| 3 | 0.5879 | 0.931 | 0.310 | 0.904 | 0.947 |
| 4 | 0.6245 | 0.845 | 0.429 | 0.936 | 0.863 |

## Confusion matrix

The evaluation output reports this confusion matrix for `xgb_class_weight_rr`.
Rows are true classes; columns are predicted classes.

| True \ Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 42,448 | 941 | 93 | 736 | 0 |
| S | 1,020 | 681 | 117 | 18 | 0 |
| V | 104 | 116 | 2,973 | 26 | 0 |
| F | 216 | 3 | 33 | 136 | 0 |
| Q | 5 | 0 | 2 | 0 | 0 |

## Interpretation

- The highest single-run macro-F1 is XGBoost with class weighting and RR features (0.6194); the CNN with class weighting has a five-seed mean of 0.6181 ± 0.0113.
- CNN class weighting has Class S sensitivity 0.877 and positive predictivity 0.386.
- XGBoost has higher Class S positive predictivity (0.391) but lower sensitivity (0.371).
- Class F positive predictivity is low across all four configurations.
- Calibration transfer was not evaluated because the Stage 2 prediction files were not found.

## Artifacts and reproducibility

The completed run wrote `artifacts/stage4/stage4_results.json` and
`artifacts/stage4/RECEIPT.json`. The notebook packaged these into
`stage4_results.zip`. The runbook says not to commit anything under `artifacts/`;
send `stage4_results.zip` to the project owner instead.

The executed notebook contains the full per-seed output. Keep it as the run record
if the project maintainers want an archived notebook, but do not duplicate the
same notebook under multiple names without a specific repository convention.

**This is the single completed DS2 evaluation. Do not rerun it.**
