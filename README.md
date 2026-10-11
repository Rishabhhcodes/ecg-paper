# ECG Arrhythmia Classification — Paper #2

Inter-patient arrhythmia classification on the MIT-BIH Arrhythmia Database,
following ANSI/AAMI EC57. A methodological sequel to our 6th-semester paper.

The question this repository exists to answer: **how much of the Class S
(supraventricular ectopic) performance comes from RR-interval timing rather
than from beat morphology or model architecture?** Every configuration is run
twice — identical model, identical folds, identical seed, with the RR features
as the only difference.

Across 13 model/strategy pairs spanning three model families, **RR features
improved both macro-F1 and Class S F1 in 26 of 26 comparisons**, with Class S
gains between 2.3× and 16.1×.

---

## Why the scores look low

Published work on MIT-BIH commonly reports accuracy above 95%. The numbers here
are much lower, for three deliberate reasons:

| | |
|---|---|
| **Inter-patient split** | No beat from a validation patient ever appears in training. Most high accuracies in the literature are intra-patient, where the model has already seen the same patient. |
| **Macro-F1, not accuracy** | DS1 is ~90% Class N. A model predicting "N" for everything scores ~0.90 accuracy and is clinically useless. Macro-F1 gives Class F the same weight as Class N. |
| **Cross-validation inside DS1** | These are development scores. DS2 is held out and has not been opened. The comparison against published figures belongs to Stage 4. |

Two minority classes are concentrated in single records, which no
patient-respecting split can spread: record 209 holds 383 of DS1's 943 S beats,
and record 208 holds 372 of its 414 F beats. **The cross-validated F score is
therefore not interpretable** — that is a property of the database, not of this
pipeline, and it explains why published F-class results vary so widely.

---

## Results so far

Best configuration per model family, 5-fold patient-grouped CV inside DS1,
macro-F1 over N/S/V/F:

| Model | Best strategy | macro-F1 | S | V | F |
|---|---|---|---|---|---|
| 1D-CNN + RR fusion | class weighting | **0.5827 ± 0.112** | 0.346 | 0.810 | 0.183 |
| XGBoost | SMOTE | 0.4994 ± 0.090 | 0.245 | 0.691 | 0.003 |
| Logistic Regression | none | 0.3691 ± 0.056 | 0.145 | 0.405 | 0.010 |

Full 26-row grid: `artifacts/grid_summary.json` after running the grid.

One result worth stating plainly. Comparing each imbalance strategy against no
correction at all, on macro-F1:

| Strategy | LR (raw waveform) | XGBoost (descriptors) | CNN (raw waveform) |
|---|---|---|---|
| class weighting | −0.071 | +0.017 | **+0.087** |
| focal loss | — | — | **+0.072** |
| SMOTE | −0.033 | +0.018 | +0.015 |
| ADASYN | −0.079 | +0.010 | +0.001 |

On morphology descriptors, reweighting the loss and resampling the data are
interchangeable. On raw waveforms, only reweighting the loss helps;
interpolation-based oversampling adds almost nothing. Logistic regression is too
weak a model for any correction to pay for itself.

These are single-seed results. Repeats across seeds are planned before any of
the smaller differences above are claimed in writing.

---

## Reproducing

Requires Python 3.11+. `pip install -r requirements.txt`.

```bash
python src/download.py          # fetch MIT-BIH into data/mitdb/  (~90 MB)
python src/build_dataset.py     # write artifacts/ds1.npz and ds2.npz
python src/verify_dataset.py    # 21 acceptance checks — all must pass
python src/run_grid.py          # the full grid
```

`run_grid.py` writes per-fold validation predictions to `artifacts/preds/` and
skips any fold whose file already exists, so an interrupted run resumes rather
than restarting. Stages 3–5 read those prediction files and never retrain.

Subsets of the grid:

```bash
python src/run_grid.py --models lr xgb     # CPU only
python src/run_grid.py --models cnn        # needs a GPU; ~1 hour on a T4
python src/run_grid.py --subset 5000       # smoke test, results not usable
```

Expected after `build_dataset.py`:

```
DS1  50,977 beats / 22 patients   N 45,824  S 943  V 3,788  F 414  Q 8
DS2  49,668 beats / 22 patients
```

Those counts are 44 below the figures in the source paper, which is exactly
accounted for: the first and last beat of every record are dropped because RR
features need both a previous and a next beat, and 22 records × 2 = 44.

---

## Two things not to change

**`src/folds.json` is the protocol.** It records which fold each patient belongs
to, and `run_grid.py` reads it rather than recomputing the split.

This is not a convenience. `StratifiedGroupKFold` is deterministic within one
scikit-learn version but **not across versions**: 1.6.1 and 1.9.1 partitioned
the same 22 patients differently from the same seed, so one half of the grid was
computed on a different split from the other. Nothing errored — the two halves
simply were not comparable. `src/make_folds.py` generated the file once; running
it again on a different scikit-learn would reintroduce the problem.

**DS2 stays sealed until Stage 4.** Not one hyperparameter, architecture choice,
loss function or threshold in this project has been selected using DS2. That is
what makes the Stage 4 number an honest estimate of generalisation rather than a
figure tuned toward.

---

## Protocol

Fixed in `src/config.py`; nothing else in the codebase re-declares a protocol
constant.

- **Split**: de Chazal DS1/DS2 — 22 development records, 22 held out
- **Excluded**: records 102, 104, 107, 217 (paced; excluded by EC57)
- **Classes**: 15 annotation symbols mapped to the five AAMI classes N, S, V, F, Q
- **Beat window**: 200 ms before and 400 ms after the R-peak → 216 samples at 360 Hz
- **R-peaks**: the database's expert annotations, never a detector — see Limitations
- **Normalisation**: per-record z-score, so no information crosses between patients
- **Evaluation**: macro-F1 over N, S, V, F. Q is excluded — DS1 holds 8 Q beats in
  total, and a single Q prediction would move the metric by more than 0.05. Q is
  excluded from the training objective as well, in both the class weights and the
  resampling strategy.
- **RR features**: `pre_rr`, `post_rr`, `local_ratio` (against the last 10 beats),
  `global_ratio` (against the whole record)

---

## Layout

```
src/
  config.py              protocol constants — the one shared file
  download.py            fetch the database
  preprocess.py          beat segmentation and RR features
  build_dataset.py       write ds1.npz / ds2.npz
  verify_dataset.py      21-check acceptance suite
  make_synthetic_records.py   synthetic records for pipeline testing
  folds.json             frozen fold assignment, per patient
  make_folds.py          generated folds.json once — do not re-run
  models.py              LR, XGBoost, 1D-CNN with RR fusion
  imbalance.py           class weights, focal loss, ADASYN / SMOTE
  run_grid.py            the experimental grid
  train_cnn.py           CNN training loop

docs/                    stage notes
notebooks/               Colab runbook and executed run records
cv_compare.py            compares fold splitters without training
fold_report.py           per-fold class distribution
```

`data/` and `artifacts/` are gitignored. Both are regenerated by the commands
above; neither is ever committed.

---

## Limitations

Stated here rather than left to be discovered.

1. **R-peaks come from expert annotations, not a detector.** A deployed system
   would need its own QRS detection and would inherit its errors, most acutely in
   the RR features. Standard practice in the inter-patient literature, but it is
   an assumption.
2. **Class F cannot be cross-validated.** Record 208 holds 90% of DS1's F beats.
3. **Class Q is excluded from evaluation.** Eight beats. Declared, not buried.
4. **Single database, single lead.** MIT-BIH, lead MLII. No cross-database
   generalisation is claimed.
5. **Twenty-two development patients is a small sample.** Fold-to-fold variance is
   large, which is why standard deviations are reported rather than point estimates.
6. **No clinical claim is made anywhere.** This produces figures for a paper.

---

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Dataset build under the inter-patient protocol | Complete |
| 2 | Model and imbalance-strategy grid, RR ablation | Complete |
| 3 | Decision layer — calibration, thresholds, reject option | Not started |
| 4 | Final evaluation on DS2 | Not started |
| 5 | Explainability and efficiency analysis | Not started |

Team workflow and branch conventions: [`CONTRIBUTING.md`](CONTRIBUTING.md).
Stage notes: [`docs/stage1.md`](docs/stage1.md), [`docs/stage2.md`](docs/stage2.md).