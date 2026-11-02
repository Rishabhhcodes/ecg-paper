# Stage 5 — Explainability and efficiency

Answers one question the rest of the project cannot. The RR ablation showed that
*removing* the timing features destroys Class S performance — 26 comparisons out
of 26. That is evidence about the feature set. It is not evidence that the
trained model **uses** them, and those are different claims.

This stage asks the model directly, then checks whether the answer can be
trusted.

**DS2 is never opened.** Everything runs inside DS1, on beats each model did not
train on. 3 seeds × 5 folds = 15 models.

---

## Method

Each beat window is 216 samples — 200 ms before the R-peak and 400 ms after. It
is split into the clinical regions already fixed in `src/config.py`, written
before this stage existed and not tuned here, plus the RR features as a fourth
unit:

| Unit | Samples | Relative to the R-peak |
|---|---|---|
| P | 0–54 | −200 ms to −50 ms |
| QRS | 54–108 | −50 ms to +100 ms |
| T | 108–216 | +100 ms to +400 ms |
| RR | — | the four timing features, which bypass the convolutions |

One unit at a time is replaced by the **training-fold mean** and the drop in the
model's probability for the true class is recorded. The baseline is the mean and
not zero: the waveform is per-record z-scored so zero is near the record mean,
but the RR features are raw seconds and ratios, where zero is not a neutral
value but an impossible heart rate.

Occlusion rather than integrated gradients: a gradient method needs a baseline,
a path, a step count and a saturation check, each a decision to defend. Occlusion
over regions fixed in advance needs none of them. The cost is resolution, and at
four units we do not need more.

---

## Finding 1 — the model leans on timing for Class S, in every run

Mean drop in P(true class) when a unit is removed, over 15 runs:

| Class | P | QRS | T | RR | Top unit |
|---|---:|---:|---:|---:|---|
| N | 0.015 | −0.058 | 0.029 | −0.057 | none — all near zero |
| **S** | −0.019 | **−0.216** | −0.007 | **+0.450** | **RR, 15/15 runs** |
| **V** | 0.059 | **+0.647** | 0.123 | 0.413 | QRS, 12/15 runs |
| F | −0.024 | 0.158 | −0.038 | 0.059 | QRS, 9/12 runs |

**RR is the top unit for Class S in every single run.** No seed, no fold
disagrees.

For Class V the model leans on QRS, which is what a ventricular ectopic beat is
defined by — a wide, abnormal complex. The attribution recovers textbook
cardiology without being told any of it.

Class N has no dominant region, which is expected: N is defined by the absence
of abnormality rather than by a feature.

### The negative number is the interesting one

For Class S, removing the QRS complex has a drop of **−0.216** — negative.
Removing morphology makes the model **more** confident the beat is S.

Taken with the RR result, the reading is that for the supraventricular ectopic
beat the morphology is not weak evidence, it is evidence *against*, and the
timing has to overcome it. That is consistent with the clinical definition: an
S beat is premature but conducted normally, so it looks like a normal beat
arriving early.

### Single-region effects

| | intact beat | remove that unit alone | restore only that unit |
|---|---:|---:|---:|
| **S**, unit RR | P(S) 0.453 | **0.003** | **0.688** |
| **V**, unit QRS | P(V) 0.789 | 0.130 | 0.160 |

Restoring RR alone, to a beat where everything else is still masked, takes P(S)
to **0.688 — higher than on the intact beat.** The timing features alone are a
stronger signal for S than the whole beat is.

---

## Finding 2 — the attribution is faithful, and this was checked

An attribution nobody checks is decoration. Units are deleted cumulatively,
most-important first, and the score tracked; then the same in a random order.
Lower deletion and higher insertion than random means the ordering is
informative.

Both the ordering and the score are restricted to one class — see
*What went wrong first* below for why that matters.

| Class | Deletion, attributed | random | wins | Insertion, attributed | random | wins |
|---|---:|---:|---:|---:|---:|---:|
| N | 0.833 | 0.878 | 13/15 | 0.938 | 0.907 | 13/15 |
| **S** | **0.061** | 0.285 | 13/15 | **0.579** | 0.279 | **15/15** |
| **V** | **0.134** | 0.291 | 13/15 | **0.472** | 0.308 | 14/15 |
| F | 0.072 | 0.112 | 5/6 | 0.171 | 0.110 | 6/6 |

For Class S the attributed ordering drops the score to **0.061** against
**0.285** for a random ordering, and insertion beats random in **15 runs out of
15**. The orderings are stable too: RR comes first for S in at least 12 of 15
runs, QRS first for V in at least 10 of 15.

**Three independent methods now agree.** The ablation says Class S collapses
without RR. The occlusion says the model leans on RR for Class S. The
faithfulness test says that attribution is not an artefact. None of the three
depends on the others.

---

## Finding 3 — efficiency

| | |
|---|---:|
| Parameters | 146,189 |
| Model size | 0.585 MB |
| CPU, batched | 0.168 ms per beat — **5,956 beats/s** |
| CPU, one beat at a time | 1.02 ms |
| GPU, batched | 0.016 ms per beat — 62,112 beats/s |

A 24-hour Holter recording holds roughly 100,000 beats. This model classifies
all of them in about **17 seconds on a single CPU core**, and a single beat takes
1 ms against the ~1,000 ms a real heartbeat lasts.

The embedded-feasibility claim rests on these numbers, and they are measured
rather than estimated.

---

## What went wrong first

The stage was run twice. The first run's attribution numbers were correct and
are unchanged here; its **faithfulness test was invalid**, for two compounding
reasons, both in how the test was written rather than in the data.

**The ordering came from the pooled row.** Units were ranked by the attribution
averaged over all beats. Class N is 89% of DS1 and its attributions are near
zero, so the pooled ranking was noise: 15 runs produced nine different
orderings, and the test then compared noise against random.

**The deletion curve was U-shaped.** With every region masked the model falls
back on its prior, which is class N. Pooled P(true class) therefore *recovered*
at the fully masked end — the curve finished higher than it started in 12 of 15
runs — so its area measured almost nothing.

The result was deletion 0.801 against 0.827 for random: no separation, and no
way to tell whether that meant the attribution was uninformative or the test
was.

Restricting both the ordering and the score to one class fixes both at once: the
ordering comes from a signal that is actually there, and masking everything
drives P(class) toward that class's prior, which for a minority class is near
zero. The corrected test separates cleanly, as the table above shows.

The fix was validated before being run on real data, on a synthetic set with a
known answer planted in it — Class S made to depend on an RR feature and Class V
on the QRS region. The corrected test recovered both and stayed flat on the two
classes with nothing planted.

---

## Limitations

1. **Four units is coarse.** Occlusion over fixed regions says which region
   matters, not which samples within it. A per-sample method would say more and
   would need its own defence.
2. **Occlusion measures one unit at a time.** If two regions carry the same
   information, removing either alone shows little, and this design would miss
   it. The deletion curves partly cover this, since they remove units
   cumulatively.
3. **The mean baseline is a choice.** A mean beat is not a physiologically
   meaningful signal; it is a neutral one. Other baselines would give other
   numbers, though the ordering is unlikely to change.
4. **Class F has 12 usable runs, not 15.** Three folds held too few F beats to
   measure, which is the same concentration problem declared in Stage 1.

---

## Reproducing

```bash
python src/stage5_explain.py --strategy class_weight --seeds 0 1 2
```

Needs `artifacts/ds1.npz` and a GPU; about 35 minutes. Writes
`artifacts/stage5/` with the results JSON and two figures.

Record of this run: [`stage5_results.json`](stage5_results.json), and the
executed notebook under `notebooks/runs/`.