# Stage 4 — The single evaluation on DS2

Four configurations, locked on Stage 2 evidence, trained on DS1 and applied
once to the 22 held-out patients. No model choice, loss function or threshold
anywhere in Stages 1–3 was selected using DS2.

Run: 22 October 2026, commit `1d69436ec1`, 9.2 minutes on a T4.
Dataset verification: **21 checks passed, 0 failed.** DS1 and DS2 share no
patient.

---

## Declare this first: DS2 was opened twice

The evaluation was run on 21 October and again on 22 October. The receipt
records only the second, and says "first opening".

**Why the guard missed it.** `stage4_final.py` writes
`artifacts/stage4/RECEIPT.json` on completion and refuses to run again if it
finds one. The receipt lives in the Colab session's filesystem, which is
destroyed when the session ends. The second run started in a fresh session, found
no receipt, and recorded itself as the first. The lock works within a session and
cannot survive one — a design fault in this script, not a lapse by the person
running it.

**Why it does not invalidate the result.** Nothing was changed between the two
runs. The proof is in the numbers: XGBoost and logistic regression are
deterministic and reproduced **exactly**, to four decimal places, in both runs.
Only the CNN moved, and only by the amount Stage 2's noise floor predicts.

| Configuration | 21 Oct | 22 Oct | |
|---|---:|---:|---|
| CNN + class weighting | 0.6012 | 0.6181 | differ by 0.0169 |
| CNN + focal loss | 0.5959 | 0.5837 | differ by 0.0122 |
| XGBoost + class weighting | 0.6194 | 0.6194 | identical |
| Logistic regression | 0.5531 | 0.5531 | identical |

The two CNN results sit 0.0085 either side of their mean against a measured
run-to-run noise floor of ±0.0114 (Stage 2). Neither run was selected for being
better — the first was simply never saved, because the Colab session ended before
anything was downloaded.

**What this accidentally bought us.** Two independent executions of the entire
evaluation. That is an external reproducibility check nobody planned, and it
passed: the deterministic half is bit-identical and the stochastic half moves
within its measured bound.

The committed numbers below are the 22 October run. The 21 October figures are
kept in the table above so that both are on the record.

---

## DS2 composition

| Class | Beats |
|---|---:|
| N | 44,218 |
| S | 1,836 |
| V | 3,219 |
| F | 388 |
| Q | 7 |

---

## Results

`Se / +P` is sensitivity and positive predictivity, the convention in the
inter-patient literature. The bracket on macro-F1 is **half the range across
five seeds**, not a standard deviation.

| Configuration | Runs | macro-F1 | N Se / +P | S Se / +P | V Se / +P | F Se / +P |
|---|---:|---:|---|---|---|---|
| CNN + class weighting + RR | 5 | 0.6181 ±0.0113 | 0.885 / 0.991 | **0.877 / 0.386** | 0.900 / 0.881 | 0.428 / 0.081 |
| CNN + focal loss + RR | 5 | 0.5837 ±0.0400 | 0.775 / 0.995 | 0.872 / 0.368 | 0.904 / 0.842 | 0.701 / 0.049 |
| XGBoost + class weighting + RR | 1 | **0.6194** | 0.960 / 0.969 | 0.371 / 0.391 | 0.924 / 0.924 | 0.351 / 0.148 |
| Logistic regression, none, + RR | 1 | 0.5531 | 0.958 / 0.955 | 0.344 / 0.567 | 0.719 / 0.736 | 0.186 / 0.069 |

Per-class F1, derived from the same numbers:

| Configuration | N | S | V | F |
|---|---:|---:|---:|---:|
| CNN + class weighting | 0.935 | **0.536** | 0.890 | 0.136 |
| CNN + focal loss | 0.871 | 0.518 | 0.872 | 0.092 |
| XGBoost + class weighting | 0.964 | 0.381 | 0.924 | 0.208 |
| Logistic regression | 0.956 | 0.428 | 0.727 | 0.101 |

---

## Finding 1 — macro-F1 cannot separate two models that are not comparable

XGBoost scores 0.6194 and the CNN 0.6181. The gap is **0.0013 against a noise
floor of ±0.0113** — they are tied, and no amount of further seeds would
separate them.

On the class this project exists to detect they are not close at all:

| | catches | misses, of 1,836 S beats |
|---|---:|---:|
| CNN + class weighting | **87.7%** | **226** |
| XGBoost + class weighting | 37.1% | 1,155 |

XGBoost misses five times as many supraventricular ectopic beats and scores the
same. It recovers the difference on Class N, where its sensitivity is 0.960
against the CNN's 0.885 — and Class N is 89% of the set, so being slightly better
at the easy class is worth as much in a four-class mean as being twice as good at
the hard one.

**This is the result worth reporting.** A macro-averaged F1 is the standard
summary in this literature precisely because it is supposed to stop the majority
class dominating. On this problem it still does not separate a model that finds
the target class from one that misses two thirds of it. Any paper that ranks
models on macro-F1 alone can reach the wrong conclusion, and here is a worked
case.

Reported alongside macro-F1, per-class sensitivity makes the difference
immediately visible — which is why the AAMI convention asks for it.

---

## Finding 2 — the model generalises exactly as the development scores predicted

DS2 macro-F1 is **higher** than the DS1 cross-validated figure for every
configuration. That ordinarily signals a leak, so it was checked.

The cause is class composition, not the model:

**Class S prevalence is 1.85% in DS1 and 3.70% in DS2 — exactly twice.**
Precision rises with prevalence for a fixed classifier, and precision feeds F1,
which feeds the macro average.

Re-scoring the same classifier at DS1's prevalence, holding its sensitivity and
false-positive rate fixed:

| | DS2 Class S F1 | re-scored at DS1 prevalence | DS1 CV said |
|---|---:|---:|---:|
| CNN + class weighting | 0.536 | **0.372** | **0.376** |
| CNN + focal loss | 0.518 | 0.354 | 0.407 |

For the configuration we report, the corrected figure lands within 0.004 of the
cross-validated one. **The CNN performs on unseen patients exactly as DS1
cross-validation predicted — no better.** The apparent improvement belongs to the
test set's composition.

Stated plainly in the paper, this is a stronger claim than a raw DS2 number: the
development protocol estimated generalisation correctly.

---

## Finding 3 — Stage 2's refusals held on unseen data

Stage 2 concluded that class weighting and focal loss cannot be separated. On
DS2:

| Run | gap | combined noise | verdict |
|---|---:|---:|---|
| 21 Oct | +0.0053 | ±0.0766 | not rankable |
| 22 Oct | +0.0344 | ±0.0513 | not rankable |

Both runs agree, and both agree with Stage 2. A conclusion we declined to draw on
development data would still have been wrong on the test set — which is the
argument for the noise-floor rule in the first place.

---

## Finding 4 — Class F is unusable, as predicted in Stage 1

Positive predictivity for Class F is 0.049 to 0.148 across all four
configurations. The CNN with focal loss reaches 0.701 sensitivity at 0.049
precision: of every twenty beats it calls F, one is.

The confusion matrix for XGBoost shows 736 Class N beats predicted as F — nearly
twice the 388 real F beats in the whole set.

This was stated as a limitation in Stage 1, before any model was trained: record
208 holds 90% of DS1's F beats, so no patient-respecting split can give a model
enough of them to learn from. The DS2 result confirms the prediction rather than
discovering a surprise.

### Confusion matrix — XGBoost + class weighting

Rows are true classes, columns predicted.

| True \ Pred | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| **N** | 42,448 | 941 | 93 | 736 | 0 |
| **S** | 1,020 | 681 | 117 | 18 | 0 |
| **V** | 104 | 116 | 2,973 | 26 | 0 |
| **F** | 216 | 3 | 33 | 136 | 0 |
| **Q** | 5 | 0 | 2 | 0 | 0 |

---

## Limitations of this stage

Four, all declared rather than discovered.

1. **The one-shot lock does not survive a session reset.** Covered at the top.
   A durable version would keep the receipt outside the run environment — in the
   repository, or as a value the runner must paste in. The result stands because
   the deterministic models prove nothing changed between runs, but the guard
   should not be relied on as written.

2. **LR and XGBoost saw more training data here than in Stage 2.** The CNN holds
   one DS1 fold out for early stopping, so it trains on 4/5 of DS1 in both
   stages and its DS1→DS2 comparison is like for like. LR and XGBoost need no
   early-stopping set and were trained on all of DS1, so their larger jumps
   (+0.12 and +0.18 on macro-F1) are partly extra data, not only generalisation.
   No conclusion in this document rests on those two.

3. **DS2 probabilities were not saved.** `stage4_final.py` writes metrics and
   confusion matrices, not the per-beat outputs. Any further DS2 analysis —
   calibration transfer, a reject curve, a threshold sweep — would require
   opening DS2 again. It should not be opened again.

4. **Calibration transfer was not measured.** The Stage 2 prediction files were
   not present in the session, so no DS1-only temperature could be fitted. This
   is the one planned result the stage did not deliver, and recovering it would
   cost a third opening. It is not worth it.

---

## Reproducing

```bash
python src/stage4_final.py --confirm
```

Requires `artifacts/ds1.npz` and `artifacts/ds2.npz` and a GPU. The
configuration list is hardcoded in the script and has no flag, because a flag
would allow several attempts and reporting the best.

Record of this run: [`stage4_results.json`](stage4_results.json),
[`stage4_receipt.json`](stage4_receipt.json), and the executed notebook at
`notebooks/runs/stage4_run_2026-10-22.ipynb`.

**Do not run this again.** Everything the paper needs is in those three files.