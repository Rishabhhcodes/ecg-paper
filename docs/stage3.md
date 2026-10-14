# Stage 3 — The Decision Layer (calibration and the reject option)

Boxes 12–15. Turns the saved Stage 2 predictions into calibrated
probabilities and reports what a reject option would buy.

**Nothing is retrained and DS2 is never opened.** This stage reads the
per-fold validation predictions `run_grid.py` already wrote. No GPU, no
`ds1.npz`, no PyTorch.

## Run it

```bash
pip install numpy scipy scikit-learn matplotlib
python src/stage3_decision.py
```

Under a minute on a laptop CPU. Flags: `--model`, `--strategy`, `--rr`,
`--seeds`. By default it analyses `cnn / focal / rr=1` and uses every
`artifacts/preds*` directory it finds.

## What it needs

| Input | Where from |
|---|---|
| `src/config.py` | the repository |
| `artifacts/preds/{model}_{strategy}_{rr}_f{0..4}.npz` | `run_grid.py` |
| `artifacts/preds_seed{1..4}/…` | the multi-seed repeat (optional but expected) |

Five directories, five fold files each. Each seed is one repeat of the
same experiment; together they are what tells us how much of any
difference is training noise.

## What comes out

```
artifacts/stage3/
  stage3_results.json    every number in the figures
  reliability.png        calibration before and after, with ECE
  arc.png                the Accuracy-Reject Curve
  pr_class_s.png         Class S precision-recall
```

---

## Finding 1 — calibration works, in the opposite direction to the textbook

A network's softmax output looks like a probability and usually is not
one. Temperature scaling (Guo et al., ICML 2017) fits a single scalar `T`
by minimising negative log-likelihood and divides the logits by it.

| Quantity | Before | After | |
|---|---|---|---|
| Expected Calibration Error | 0.1204 | 0.0724 | **−40%** |
| Temperature `T` | — | **0.73 ± 0.14** | below 1 |
| Predictions changed | — | 0 | asserted every run |

`T < 1` means the model was **under**confident and is being sharpened —
the opposite of Guo et al.'s headline, which is about cross-entropy
models. For focal loss this is the documented direction: Mukhoti et al.
(NeurIPS 2020) show focal loss raises the entropy of the predicted
distribution, "thereby preventing the model from becoming overconfident".

Their optimal temperatures mostly lie between 0.9 and 1.1, so 0.73 is
further from 1 than they typically observe. We do not have an
explanation for that, and the paper says so.

### Two things this stage proves about itself

**`T` is cross-fitted.** For each fold, `T` is fitted on the pooled
validation predictions of the *other four* folds. Fitting and scoring on
the same beats would be the first thing a reviewer asks about. Because
the folds are patient-disjoint, the calibration is inter-patient too.

**No Stage 2 number can move.** A single positive temperature is
monotonic, so it cannot change which class has the highest probability,
and every Stage 2 figure is an argmax. The script asserts this rather
than relying on the argument.

### Probabilities, not logits — and why that is exact

`run_grid.py` saves `proba`, and temperature scaling is defined on
logits. This is not an approximation. Since `p = softmax(z)`:

```
log(p) = z − logsumexp(z)
```

so `log(p)` recovers `z` up to one additive constant shared by all
classes, and softmax is invariant to exactly that constant:

```
softmax(log(p)/T) ≡ softmax(z/T)
```

Holds for all three model families — multinomial logistic regression and
XGBoost's `multi:softprob` are softmax outputs as well. Float32
probabilities can underflow to zero for very confident predictions, so a
floor of 1e-12 is applied and the number of beats it touched is printed.

---

## Finding 2 — the reject option does not help

Letting the model abstain on its least confident beats is the standard
next move. Sweeping the rejection rate from 0% to 50%:

| Rejected | macro-F1 change | seeds up | Class S change | seeds up |
|---|---|---|---|---|
| 10% | +0.002 | 3/5 | **−0.037** | 1/5 |
| 20% | +0.006 | 4/5 | **−0.069** | 0/5 |
| 35% | +0.007 | 4/5 | **−0.092** | 0/5 |
| 50% | −0.003 | 2/5 | **−0.058** | 0/5 |

Each figure is that seed measured against **its own** 0% point. That
matters: the question is whether rejecting improves a given model, which
is a paired comparison, not whether two models differ. The spread
*between* seeds is much larger than the paired gain and is the wrong
thing to test against.

**Class S falls at every rejection rate, in every seed, under both ways
of aggregating it.** macro-F1 does not move.

The mechanism is structural: the beats the model is least confident about
are disproportionately the minority-class beats, so any confidence filter
removes exactly what the work is trying to detect.

> The Accuracy-Reject Curve is conventionally drawn on accuracy, which
> rises monotonically under rejection. **On an imbalanced problem with a
> macro metric it does not.** Anyone adopting a reject option for
> rare-event detection should expect this.

## Finding 3 — per-class thresholds are not worth their cost

The best achievable point on the Class S precision-recall curve beats the
plain argmax by **+0.011** — and that point is chosen on the same data it
is measured on, so the achievable gain is smaller still. Four tuned
numbers, each needing justification, for less than the noise floor. Not
pursued.

---

## Nothing here picks an operating point

No threshold, no rejection rate and no precision floor is selected
anywhere in this stage. Chow (1970) shows the rejection threshold *is*
the cost assigned to a rejection, so the quantity one picks is a cost
ratio rather than a rate. We report the whole curve instead and leave the
operating point to the reader — which leaves nothing arbitrary in this
stage for a reviewer to attack.

Class S is reported the same way, as a precision-recall curve rather than
at a chosen precision floor.

---

## Two defects this stage surfaced

Both silent, both found by cross-check rather than by an error.

**A partial grid run had clobbered the summary.** Stage 3 and Stage 2
disagreed about Class S F1 by 0.02. The cause was seed 0's ten CNN rows
missing from `grid_summary.json` after an unrelated `--models lr xgb` run
rewrote it. See the end of [`stage2.md`](stage2.md).

**Two reductions were mistaken for one.** `run_grid.py` averages macro-F1
over folds but computes `per_class_f1` on the pooled beats. Applying the
fold average to Class S as well **reversed the sign** of the
reject-option result — same data, same code, opposite conclusion.

Across seeds the pooled macro-F1 moves by ±0.0325 while the mean over
folds moves by ±0.0087. Pooling mixes folds whose class balance is very
different, and record 208 then dominates the F score. The choice of
reduction is not presentational.

### The guard that now exists

At 0% rejection nothing has been rejected, so the curve must start
exactly on the Stage 2 numbers:

```
AssertionError: seed 0: the reject curve starts at f1_S=0.7982
but the Stage 2 reduction gives 0.7993. The curve and the
cross-check are not computing the same thing.
```

Each metric uses the same reduction `run_grid.py` uses for it, and the
script checks that it has. The bug was deliberately reintroduced once to
confirm the assertion fires.

---

## References

- Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). *On Calibration
  of Modern Neural Networks.* PMLR 70:1321–1330.
- Chow, C.K. (1970). *On optimum recognition error and reject tradeoff.*
  IEEE Trans. Inf. Theory 16(1):41–46.
- Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P.H.S.,
  Dokania, P.K. (2020). *Calibrating Deep Neural Networks using Focal
  Loss.* NeurIPS 33.
- Hendrickx, K., Perini, L., Van der Plas, D., Meert, W., Davis, J.
  *Machine Learning with a Reject Option: A survey.* arXiv:2107.11277.

---

Stage 4 is the single evaluation on DS2. It has not been opened, and
nothing in this stage has been tuned toward it.