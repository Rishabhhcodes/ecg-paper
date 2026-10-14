# ECG Arrhythmia Classification — Paper #2

Inter-patient arrhythmia classification on the MIT-BIH Arrhythmia Database,
following ANSI/AAMI EC57. A methodological sequel to our 6th-semester paper.

The question this repository exists to answer: **how much of the Class S
(supraventricular ectopic) performance comes from RR-interval timing rather
than from beat morphology or model architecture?** Every configuration is run
twice — identical model, identical folds, identical seed, with the RR features
as the only difference.

Across 13 model/strategy pairs spanning three model families, **RR features
improved both macro-F1 and Class S F1 in 26 of 26 comparisons** — every one of
them by a margin larger than the run-to-run noise we measured across five
seeds.

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
macro-F1 over N/S/V/F. The CNN was repeated across five seeds and is reported as
the mean, with half the range across seeds as the spread. Logistic regression is
deterministic, so its figures are exact. XGBoost was not repeated; its spread is
assumed small rather than measured, and nothing below rests on an XGBoost
ranking.

| Model | Best strategy | macro-F1 | Class S F1 |
|---|---|---|---|
| 1D-CNN + RR fusion | class weighting / focal — **indistinguishable** | 0.578 / 0.574 | 0.376 / 0.407 |
| XGBoost | SMOTE / class weighting / ADASYN — within 0.008 | 0.499 | 0.245 |
| Logistic Regression | none | 0.369 | 0.145 |

Full 26-row grid: `artifacts/grid_summary*.json`.
Cross-seed analysis: `python src/seed_report.py`.

Comparing each imbalance strategy against no correction at all, on macro-F1,
with the RR features present:

| Strategy | LR (raw waveform) | XGBoost (descriptors) | CNN (raw waveform) |
|---|---|---|---|
| class weighting | −0.071 | +0.017 | **+0.055** |
| focal loss | — | — | **+0.052** |
| SMOTE | −0.033 | +0.018 | −0.010 *(n.s.)* |
| ADASYN | −0.079 | +0.010 | −0.030 *(n.s.)* |

Only the CNN column has measured noise behind it. Its two bold entries exceed
that noise; the two marked *n.s.* do not, and are reported as no effect.

What survives: **on raw waveforms, reweighting the loss works and resampling
does not.** Class weighting and focal loss each gain about 0.05; SMOTE and
ADASYN gain nothing. Measured against class weighting rather than against
nothing, ADASYN is behind by 0.085 on macro-F1 and 0.199 on Class S, both well
outside the noise. On morphology descriptors the picture is different — there
every strategy helps a little and they are interchangeable — but those rows are
single-seed and nothing here leans on them.

We cannot say ADASYN is worse than doing nothing. An earlier version of this
file did; at five seeds that gap (−0.030) sits inside the noise (±0.037). What
can be said is that it is clearly worse than the reweighting strategies, and
that the same sampler on morphology descriptors does not behave this way. The
likely reason is that ADASYN concentrates synthetic samples near the decision
boundary, which on raw ECG is where the ambiguous and mislabelled beats sit.

Logistic regression is too weak a model for any correction to pay for itself.

### What cannot be ranked

The CNN was run five times with everything fixed except its own initialisation.
Measured spread: **macro-F1 ±0.0114** (worst configuration ±0.0249), **Class S
F1 ±0.0496** (worst ±0.1650). A difference smaller than that is not a finding.

| Comparison | Metric | Gap | Noise | Verdict |
|---|---|---|---|---|
| CNN class weighting vs focal | macro-F1 | +0.0038 | ±0.0178 | tie |
| CNN class weighting vs focal | Class S | −0.0307 | ±0.0897 | tie |
| CNN SMOTE vs ADASYN | macro-F1 | +0.0203 | ±0.0313 | tie |
| CNN SMOTE vs ADASYN | Class S | +0.0554 | ±0.1207 | tie |
| CNN class weighting vs SMOTE | macro-F1 | +0.0649 | ±0.0287 | **class weighting** |
| CNN class weighting vs SMOTE | Class S | +0.1438 | ±0.0760 | **class weighting** |

**Class weighting and focal loss cannot be separated**, on either metric. An
earlier version of this file claimed focal loss was ahead on Class S and
recommended it on that basis. That claim came from a four-seed average — one
seed's CNN rows had been dropped from the summary without anything erring — and
from a comparison made by hand rather than through the noise rule, which at the
time only checked macro-F1. With all five seeds the gap is 0.031 against
±0.090. Both configurations are reported; neither is preferred.

---

## Stage 3 — the decision layer

`src/stage3_decision.py` reads the saved validation predictions and never
retrains. One minute on a CPU, no GPU and no dataset needed.

**Calibration helps.** Temperature scaling (Guo et al., ICML 2017) fits a single
scalar `T` per fold, on the other four folds, so no fold is calibrated on
itself. **ECE 0.120 → 0.072.**

`T = 0.73 ± 0.14`, which is *below* 1: the model was under-confident and is
being sharpened — the opposite of Guo et al.'s headline finding. That is the
documented direction for focal loss. Mukhoti et al. (NeurIPS 2020) show focal
loss raises the entropy of the predicted distribution, "thereby preventing the
model from becoming overconfident". Their optimal temperatures mostly sit
between 0.9 and 1.1, so 0.73 is further from 1 than they typically observe.

Because `T` cannot change which class has the highest probability, **no Stage 2
number moves**. The script asserts this rather than assuming it, and separately
asserts that the reject curve at 0% rejection reproduces the Stage 2 figures.

**The reject option does not help.** Sweeping the rejection rate from 0% to
50%, macro-F1 stays flat (+0.006 at best) and **Class S F1 falls at every
rejection rate, in every seed, under both ways of aggregating it.**

The reason is structural: the beats the model is least confident about are
disproportionately the minority-class beats. Any confidence-based filter removes
exactly what the work is trying to detect. Worth stating, because the
Accuracy-Reject Curve is conventionally drawn on accuracy, which rises
monotonically under rejection — **on an imbalanced problem with a macro metric
it does not.**

No operating point is chosen anywhere. Chow (1970) shows the rejection threshold
*is* the cost assigned to a rejection, so what one picks is a cost ratio rather
than a rate; here the whole curve is reported instead, leaving nothing arbitrary
to defend. The same applies to Class S, reported as a precision-recall curve
rather than at a chosen precision floor.

Per-class thresholds were considered and dropped. The best point on the Class S
PR curve beats the argmax by **+0.011**, and that point is selected on the same
data it is measured on, so the achievable gain is smaller still. Four tuned
numbers for less than the noise floor is not a trade worth defending.

Figures in `artifacts/stage3/`: `reliability.png`, `arc.png`, `pr_class_s.png`.

---

## Reproducing

Requires Python 3.11+. `pip install -r requirements.txt`.

```bash
python src/download.py          # fetch MIT-BIH into data/mitdb/  (~90 MB)
python src/build_dataset.py     # write artifacts/ds1.npz and ds2.npz
python src/verify_dataset.py    # 21 acceptance checks — all must pass
python src/run_grid.py          # the full grid
python src/stage3_decision.py   # calibration and reject curve
```

`run_grid.py` writes per-fold validation predictions to `artifacts/preds/` and
skips any fold whose file already exists, so an interrupted run resumes rather
than restarting. Stages 3–5 read those prediction files and never retrain.

Subsets of the grid:

```bash
python src/run_grid.py --models lr xgb     # CPU only
python src/run_grid.py --models cnn        # needs a GPU; ~1 hour on a T4
python src/run_grid.py --seed 1            # a repeat, into artifacts/preds_seed1/
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

## Three things not to change

**`src/folds.json` is the protocol.** It records which fold each patient belongs
to, and `run_grid.py` reads it rather than recomputing the split.

This is not a convenience. `StratifiedGroupKFold` is deterministic within one
scikit-learn version but **not across versions**: 1.6.1 and 1.9.1 partitioned
the same 22 patients differently from the same seed, so one half of the grid was
computed on a different split from the other. Nothing errored — the two halves
simply were not comparable. `src/make_folds.py` generated the file once; running
it again on a different scikit-learn would reintroduce the problem.

**The prediction files are the result; the summaries are a view of them.**
`run_grid.py` rebuilds its summary from scratch on every run and writes the
whole file, so a partial run such as `--models lr xgb` replaces a 26-row
`grid_summary.json` with a 16-row one and the CNN rows are gone. That happened.
Nothing errored, the predictions were untouched, and `seed_report.py` went on
averaging the CNN over four seeds while reporting that it had five.

```bash
python src/verify_summary.py            # every row against the .npz files
python src/rebuild_summary.py --write   # regenerate from the .npz files
```

Neither can train anything. When a summary and the predictions disagree, the
predictions win.

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

Two reductions appear throughout and are not interchangeable. `macro_f1_mean` is
the **mean over the five folds**; `per_class_f1` is computed on the **pooled**
beats. Across seeds the first moves by ±0.0087 and the second by ±0.0325 —
pooling mixes folds whose class balance is very different, and record 208 then
dominates the F score. Stage 3 matches each metric to its Stage 2 reduction, and
asserts that it has.

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
  seed_report.py         cross-seed means, noise floor, rankability
  verify_summary.py      summaries checked against the prediction files
  rebuild_summary.py     summaries regenerated from the prediction files
  stage3_decision.py     calibration and the reject curve

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
4. **Only the CNN was repeated across seeds.** Logistic regression is
   deterministic, so its figures are exact; XGBoost was not repeated, so no
   XGBoost ranking in this file is backed by a measured noise floor.
5. **Single database, single lead.** MIT-BIH, lead MLII. No cross-database
   generalisation is claimed.
6. **Twenty-two development patients is a small sample.** Fold-to-fold variance is
   large, which is why spreads are reported rather than point estimates.
7. **No clinical claim is made anywhere.** This produces figures for a paper.

---

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Dataset build under the inter-patient protocol | Complete |
| 2 | Model and imbalance-strategy grid, RR ablation | Complete |
| 3 | Decision layer — calibration, thresholds, reject option | Complete |
| 4 | Final evaluation on DS2 | Not started |
| 5 | Explainability and efficiency analysis | Not started |

Team workflow and branch conventions: [`CONTRIBUTING.md`](CONTRIBUTING.md).
Stage notes: [`docs/stage1.md`](docs/stage1.md), [`docs/stage2.md`](docs/stage2.md),
[`docs/stage3.md`](docs/stage3.md).

### References

- Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). *On Calibration of
  Modern Neural Networks.* PMLR 70:1321–1330.
- Chow, C.K. (1970). *On optimum recognition error and reject tradeoff.* IEEE
  Transactions on Information Theory, 16(1):41–46.
- Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P.H.S., Dokania,
  P.K. (2020). *Calibrating Deep Neural Networks using Focal Loss.* NeurIPS 33.
- Hendrickx, K., Perini, L., Van der Plas, D., Meert, W., Davis, J. *Machine
  Learning with a Reject Option: A survey.* arXiv:2107.11277.