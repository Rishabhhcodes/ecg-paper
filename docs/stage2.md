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
3 models  x  4 strategies  x  2 RR settings  x  5 folds
```

Focal loss is a neural-network objective, so it is skipped for LR and
XGBoost — the script says so rather than silently substituting something
else.

| | |
|---|---|
| Models | `lr` (Paper #1 baseline) · `xgb` (morphology + RR) · `cnn` (waveform + RR fusion) |
| Strategies | `none` · `class_weight` · `focal` · `adasyn` |
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
combination is wanted later, add it as a fifth strategy and report it as
its own row.

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
