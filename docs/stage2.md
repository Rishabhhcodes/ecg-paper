# Stage 2 — Model Training (DS1 only)

Boxes 8–11. Runs the experimental grid with patient-grouped
cross-validation and writes per-fold predictions that Stages 3–5 consume.

**DS2 is never opened by this stage.** Not for a sanity check, not for a
quick look. That discipline is the paper's central claim.

## Who runs what

Stage 2 splits cleanly across two people — different hardware, no
dependency between them.

| Part | Command | Where | Time |
|---|---|---|---|
| **A** — LR + XGBoost | `python src/run_grid.py --models lr xgb` | Laptop CPU | 30–90 min |
| **B** — CNN | `python src/run_grid.py --models cnn` | Colab GPU | 2–4 h |

Part B needs `src/train_cnn.py` and `torch`, which arrive with that
contribution. Part A runs without either — the CNN import inside
`run_grid.py` is lazy, so it is only touched when `--models cnn` is
passed.

## The grid

```
3 models  x  5 strategies  x  2 RR settings  x  5 folds   =  26 configurations
(focal is skipped for LR and XGBoost, which is why it is 26 and not 30)
```

Focal loss is a neural-network objective, so it is skipped for LR and
XGBoost — the script says so rather than silently substituting something
else.

| | |
|---|---|
| Models | `lr` (Paper #1 baseline) · `xgb` (morphology + RR) · `cnn` (waveform + RR fusion) |
| Strategies | `none` · `class_weight` · `focal` · `adasyn` · `smote` |
| RR setting | `1` = with RR features, `0` = without → this pair **is** the ablation |

## Running Part A

**1. Smoke test first — always.**

```bash
python src/run_grid.py --models lr xgb --subset 5000
rm -rf artifacts/preds artifacts/grid_summary.json
```

Ten minutes here saves an hour later. Numbers from a 5,000-beat subset
are meaningless — you are only checking that nothing crashes. Delete the
output afterwards so a smoke-test file cannot leak into real results.

**2. The real run.**

```bash
python src/run_grid.py --models lr xgb
```

**3. Check the ablation.**

```bash
python -c "
import json
s = json.load(open('artifacts/grid_summary.json'))
for r in s:
    print(f\"{r['model']:4} {r['strategy']:12} rr={int(r['use_rr'])}  \"
          f\"macroF1 {r['macro_f1_mean']:.4f}  S={r['per_class_f1']['S']:.3f}\")
"
```

Compare `rr=1` against `rr=0` for the same model and strategy. **The gap
in the S column is the result this whole stage exists to produce.**

## What to expect

Published inter-patient work lands at **S-class F1 ≈ 68–75**:

| Work | SVEB F1 | VEB F1 |
|---|---|---|
| de Chazal et al. (2004) | — (Se 75.9 %, PPV 38.5 %) | — (Se 77.7 %, PPV 81.9 %) |
| Romdhane et al. (2021) | 74.29 | 92.40 |
| Two-stage RR CNN (2022) | 68.61 | 95.99 |
| ICC-WKELM (2023) | 68.80 | 93.52 |

This table works in **both directions**. Below ~60 suggests a bug. Above
~85 suggests a leak — stop and investigate rather than celebrate.

LR will land well below this band; that is expected and is the point of
keeping it as the Paper #1 baseline.

## Resuming after an interruption

Every fold writes `artifacts/preds/{model}_{strategy}_{rr}_f{fold}.npz`
as soon as it finishes. Re-running the same command skips completed
folds, so a crash at fold 30 costs minutes, not hours.

The resume check validates that a cached fold matches the current data
split before reusing it. If you smoke-tested with `--subset` and then run
the full grid, the mismatch is detected and the fold recomputed — but
delete `artifacts/preds/` after a smoke test anyway rather than relying
on the guard.

## Merging Part A and Part B

Each person's run writes a `grid_summary.json` containing only *their*
configurations, so neither file is the complete table.

1. Both zip `artifacts/preds/` (~10 MB) and share via Drive
2. One person combines both into a single `preds/` folder
3. That person runs the **full** grid: `python src/run_grid.py`

Because every fold file already exists, the resume logic skips all
training and simply writes the complete summary. Seconds, not hours.

For this to work both people need **identical `config.py` and seed** —
`GroupKFold` is deterministic, so same data plus same settings gives the
same folds on both machines. That is what makes the two halves
comparable. Nobody edits `config.py`; nobody changes `--seed`.

## Design decisions worth defending in the paper

**GroupKFold, not KFold.** Folds are split by `patient_id`, and the
script asserts zero patient overlap in every fold before training starts.
A plain `KFold` would silently rebuild the leak Stage 1 removed — inside
the CV loop, where it is much harder to notice.

**Resampling happens inside the training fold only.** ADASYN is fitted
after the split, never before. Oversampling before splitting puts a
synthetic twin of a validation beat into training.

**ADASYN is capped at 35 % of the majority class, not full parity.**
Paper #1 oversampled to full parity and macro precision fell from 0.80 to
0.45. Partial rebalancing is the validated recipe. The cap lives in
`imbalance.CAP_RATIO`.

**Strategies are never stacked.** Focal loss up-weights hard examples;
ADASYN synthesises more samples near hard examples. Running both is the
same lever pulled twice, which is what over-corrected last time. If a
combination is wanted later, add it as its own strategy and report it as
its own row.

**SMOTE is a separate row, not a fallback.** An earlier version let a
failed ADASYN fall back to SMOTE inside the same row. A sampler being
compared must not silently become its rival; SMOTE is now its own
strategy and a failed sampler reports the failure in `meta`.

**Q is excluded from the objective, not only from the metric.** Eight Q
beats in DS1. Left in, they were oversampled to roughly 12,800 and
carried a per-sample weight 126x Class S. Both `class_weights` and
`apply_resample` are restricted to `config.EVAL_LABELS`.

**The folds are read from `src/folds.json`, never recomputed.**
`StratifiedGroupKFold` is deterministic within one scikit-learn version
and not across versions: 1.6.1 and 1.9.1 partitioned the same 22 patients
differently from the same seed. Do not run `make_folds.py` again.

**ADASYN falls back to SMOTE, then to no resampling.** ADASYN needs
enough neighbours in each minority class and genuinely fails for class F
in some folds. Which path was taken is recorded in each prediction file's
`meta` field, so Methods can state it precisely.

**XGBoost gets morphology descriptors, not 216 raw timepoints.** Trees
treat every column independently and cannot see waveform shape, so raw
samples waste the model. `models.morphology_features` supplies segment
statistics, slope, energy and a coarse downsample instead.

**Macro-F1 covers N, S, V, F only.** Q holds 8 and 7 beats after
paced-record removal — see `config.EVAL_CLASSES`. Its per-class F1 is
still reported, just not averaged in.

## Output

| File | Contents |
|---|---|
| `artifacts/preds/*.npz` | `proba` · `y_true` · `pid` · `val_idx` · `macro_f1` · `meta` |
| `artifacts/grid_summary.json` | mean ± std macro-F1 and per-class F1 per configuration |

Stages 3, 4 and 5 read **only** these files. You never retrain to redo a
metric, a threshold, or an explanation.

---

## Results

Five-fold patient-grouped CV inside DS1, macro-F1 over N/S/V/F. The CNN
was repeated across five seeds and is reported as the mean with half the
range in brackets. Logistic regression is deterministic, so its figures
are exact. **XGBoost was not repeated** — no XGBoost ranking below is
backed by a measured noise floor, and none is presented as a finding.

Regenerate everything here with `python src/seed_report.py`.

### The RR ablation

This is the paper. Every configuration was run twice, identical in every
respect except whether the four RR features are present.

**26 of 26 comparisons favour the RR features**, on both macro-F1 and
Class S F1, and every one by a margin larger than the measured noise.

| Configuration | Class S F1, no RR | with RR | Factor |
|---|---|---|---|
| `cnn none` | 0.0592 | 0.3648 | 6.2x |
| `cnn class_weight` | 0.0781 | 0.3761 | 4.8x |
| `cnn focal` | 0.0983 | 0.4069 | 4.1x |
| `xgb class_weight` | 0.0371 | 0.2603 | 7.0x |
| `lr none` | 0.0087 | 0.1446 | 16.6x |

Read the weakest model last. Logistic regression on waveform morphology
alone scores 0.009 on Class S — it essentially cannot find the class.
Four timing features take it to 0.145.

### Per model, best configuration

| Model | Best strategy | macro-F1 | Class S F1 |
|---|---|---|---|
| 1D-CNN + RR fusion | class weighting / focal — indistinguishable | 0.578 / 0.574 | 0.376 / 0.407 |
| XGBoost | SMOTE / class weighting / ADASYN — within 0.008 | 0.499 | 0.245 |
| Logistic Regression | none | 0.369 | 0.145 |

### The CNN across five seeds

| Strategy | RR | macro-F1 | Class S F1 |
|---|---|---|---|
| `class_weight` | 1 | **0.5777** (±0.0091) | 0.3761 (±0.0258) |
| `focal` | 1 | 0.5739 (±0.0087) | **0.4069** (±0.0639) |
| `none` | 1 | 0.5223 (±0.0249) | 0.3648 (±0.1650) |
| `smote` | 1 | 0.5128 (±0.0196) | 0.2323 (±0.0502) |
| `adasyn` | 1 | 0.4924 (±0.0117) | 0.1769 (±0.0705) |
| `class_weight` | 0 | 0.4161 (±0.0065) | 0.0781 (±0.0374) |
| `focal` | 0 | 0.4170 (±0.0060) | 0.0983 (±0.0350) |
| `none` | 0 | 0.4315 (±0.0054) | 0.0592 (±0.0256) |
| `smote` | 0 | 0.4180 (±0.0138) | 0.0280 (±0.0089) |
| `adasyn` | 0 | 0.4180 (±0.0084) | 0.0250 (±0.0136) |

### The noise floor

| Metric | Mean half-range | Worst configuration |
|---|---|---|
| macro-F1 | ±0.0114 | ±0.0249 |
| Class S F1 | ±0.0496 | ±0.1650 (`none` with RR) |

**Class S is four times noisier than macro-F1.** That single fact decides
which comparisons in this grid can be made at all. A GPU reorders its
arithmetic between runs; here that is enough to change which epoch early
stopping selects, and a rare-class score depends heavily on that choice.

### Reweighting the loss against resampling it

Each strategy against no correction at all, on macro-F1, RR present:

| Strategy | LR (raw waveform) | XGBoost (descriptors) | CNN (raw waveform) |
|---|---|---|---|
| class weighting | −0.071 | +0.017 | **+0.055** |
| focal loss | — | — | **+0.052** |
| SMOTE | −0.033 | +0.018 | −0.010 *(n.s.)* |
| ADASYN | −0.079 | +0.010 | −0.030 *(n.s.)* |

On raw waveforms, reweighting the loss works and resampling does not.
Measured against class weighting rather than against nothing, ADASYN is
behind by 0.085 on macro-F1 and 0.199 on Class S — both outside the
noise. On morphology descriptors every strategy helps a little and they
are interchangeable, but those rows are single-seed.

Logistic regression is too weak a model for any correction to pay for
itself.

### What cannot be ranked

A gap counts only if it exceeds the combined half-ranges of the two
configurations.

| Comparison | Metric | Gap | Noise | Verdict |
|---|---|---|---|---|
| `cnn` class_weight vs focal | macro-F1 | +0.0038 | ±0.0178 | tie |
| `cnn` class_weight vs focal | Class S | −0.0307 | ±0.0897 | tie |
| `cnn` smote vs adasyn | macro-F1 | +0.0203 | ±0.0313 | tie |
| `cnn` smote vs adasyn | Class S | +0.0554 | ±0.1207 | tie |
| `cnn` class_weight vs smote | macro-F1 | +0.0649 | ±0.0287 | **class_weight** |
| `cnn` class_weight vs smote | Class S | +0.1438 | ±0.0760 | **class_weight** |
| `xgb` smote vs class_weight | both | small | — | one seed — not rankable |

**Class weighting and focal loss cannot be separated**, on either metric.
Both are reported; neither is preferred.

An earlier revision recommended focal loss on the strength of a Class S
gap of +0.0434 against ±0.0241. That came from a four-seed average — one
seed's CNN rows had been dropped from `grid_summary.json` by an unrelated
partial run — and from a comparison made by hand, because the automated
rule only checked macro-F1. Both gaps are closed: `seed_report.py` now
checks Class S too, and names any configuration missing a seed before it
prints anything.

### Expectation against outcome

| Expected | Outcome |
|---|---|
| RR features would help, modestly | 26 of 26, Class S gains of 4x to 16x. The largest effect in the study |
| A best imbalance strategy would emerge | Class weighting and focal loss are indistinguishable. No preference is defensible |
| ADASYN, the more sophisticated sampler, would beat SMOTE | Indistinguishable from SMOTE, and clearly behind both reweighting strategies on raw waveforms |
| Run-to-run noise around 0.003 | ±0.0114 on macro-F1, ±0.0496 on Class S. The 0.003 estimate came from a single accidental repeat |

## Summaries are derived; predictions are the result

`run_grid.py` rebuilds its summary from scratch on every run and writes
the whole file. A partial run such as `--models lr xgb` therefore
**replaces** a 26-row `grid_summary.json` with a 16-row one, and the CNN
rows are gone. That happened. Nothing errored, the per-fold predictions
were untouched, and `seed_report.py` went on averaging the CNN over four
seeds while reporting that it had five.

```bash
python src/verify_summary.py            # every row against the .npz files
python src/rebuild_summary.py           # dry run — says what is missing
python src/rebuild_summary.py --write   # regenerate from the .npz files
```

Neither can train anything; they only read `artifacts/preds*/`. When a
summary and the predictions disagree, the predictions win.

Stage 3 is in [`stage3.md`](stage3.md).