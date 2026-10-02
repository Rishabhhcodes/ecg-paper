# Stage 1 — Data Preparation (patient-independent protocol)

Boxes 1–7 of the framework. Produces `ds1.npz` / `ds2.npz`, which are the
only inputs Stage 2 is allowed to read.

## Run it

```bash
pip install -r requirements.txt

python src/download.py         # ~90 MB from PhysioNet, run once
python src/build_dataset.py    # builds artifacts/ds1.npz, ds2.npz
python src/verify_dataset.py   # acceptance tests — must pass before Stage 2
```

No internet, or want a fast iteration loop? Test the plumbing on fake data:

```bash
python src/make_synthetic_records.py data/synthetic
python src/build_dataset.py data/synthetic
```

Synthetic output is for testing the code path only. **No number from it
ever goes in the paper.** Delete `artifacts/` and `data/synthetic/`
afterwards so nobody confuses the two.

## What comes out

| File | Contents |
|---|---|
| `ds1.npz` | `X (n,216)` beats · `rr (n,4)` timing · `y (n,)` labels · `pid (n,)` patient |
| `ds2.npz` | same, from 22 *different* patients |
| `manifest.json` | per-record counts, class distribution, config echo — paste into the paper's Methods |

## Expected output

On real MIT-BIH:

```
             total       N       S       V       F       Q
DS1         50,977  45,824     943   3,788     414       8
DS2         49,668  44,218   1,836   3,219     388       7
```

Those totals sit 44 beats below the commonly published figures — exactly
22 records × 2 edge beats, dropped because the first and last beat of
each record have no pre-RR or post-RR. That arithmetic is the sanity
check: if your totals are ~19,000 you are still on synthetic data.

## Design decisions, and why

**Records.** 48 total, minus paced 102/104/107/217 as ANSI/AAMI EC57
requires, leaves 44. DS1 (22) trains, DS2 (22) tests. Lists are from
de Chazal, O'Dwyer & Reilly (2004), IEEE TBME 51(7).

**R-peaks come from the database's expert annotations, not a detector.**
Detector error would confound the imbalance results, and it is a separate
variable we are not studying. Say this explicitly in Methods — reviewers
ask.

**Window: 200 ms before, 400 ms after R (216 samples at 360 Hz).**
Keeps the preceding P wave and the following T wave inside the window,
which Stage 5 needs to aggregate attributions into P/QRS/T regions.
Boundaries live in `config.SEGMENTS`.

**Normalisation is per-record z-score.** Statistics never cross a patient
boundary, so it is leak-free by construction. It also removes the
per-patient amplitude and baseline offset that a model would otherwise
use to identify the patient rather than the disease.

**Four RR features.** `pre_rr`, `post_rr`, `local_ratio` (vs the last 10
beats), `global_ratio` (vs the whole record). Class S is defined by
prematurity, and a window cropped around a beat's own R-peak discards
exactly that. Without these, S is unidentifiable no matter which model is
used. The two ratios are relative to the patient's own rhythm, so they
help across patients too.

**First and last beat of each record are dropped** — no pre-RR or post-RR
exists for them.

**`pid` must survive to the end of the project.** Stage 2's `GroupKFold`
depends on it. A plain `KFold` would silently re-create the leak this
whole stage exists to remove.

**Q is kept in the data but excluded from metrics.** After paced-record
removal it holds 8 and 7 beats. See the note in `src/config.py`.

## Before you trust any output

`verify_dataset.py` runs three tiers:

- **PROTOCOL** — patient disjointness, paced exclusion, record lists.
  A failure here invalidates everything downstream.
- **SHAPE / NORMALISATION** — array alignment, window length, no NaNs.
- **RR DIAGNOSTIC** — S beats must show a *lower* local RR ratio than N
  beats, since S beats are premature by definition. If this fails, the RR
  features are wired wrong and Class S will stay broken regardless of
  architecture.

Reference values from a verified run: separation **+0.231 (DS1)** and
**+0.318 (DS2)**. Positive is what matters; the exact figure will vary
slightly.

The SCALE check compares totals against ~51,000 (DS1) and ~49,700 (DS2).
Warnings there are expected on synthetic data and meaningful on real data.

## Sign-off before Stage 2

- [ ] `verify_dataset.py` passes all PROTOCOL and SHAPE checks on real data
- [ ] Guide has signed off on the AAMI mapping table in `config.py`
- [ ] `manifest.json` class distribution recorded in the shared results sheet
- [ ] `ds1.npz` / `ds2.npz` uploaded to the shared Drive folder
- [ ] Repo pushed; `artifacts/` gitignored (regenerate, don't commit)

## Next

Stage 2 reads `ds1.npz` only. `ds2.npz` stays sealed until the single
final evaluation run — no peeking for hyperparameters, no "just checking"
on the test set. That discipline is the paper's main claim.
