"""
Box 10 — the five imbalance strategies.

Rules enforced here, because getting these wrong is how Paper #1's
precision collapsed and how leaks get reintroduced:

  1. Resampling is fitted on the TRAINING FOLD ONLY. Never on the
     validation fold, never before the split. A synthetic twin of a
     validation beat appearing in training is a silent leak.
  2. Oversampling is capped at CAP_RATIO of the majority class, not
     full parity. Paper #1 oversampled to full parity (72,471 per
     class) and macro precision fell 0.80 -> 0.45. Partial rebalancing
     is the validated recipe.
  3. Strategies are evaluated SEPARATELY, never stacked. Focal loss
     up-weights hard examples; ADASYN synthesises more samples near
     hard examples. Same lever pulled twice.
  4. ADASYN and SMOTE are two SEPARATE rows, and neither falls back to
     the other. They are being compared against each other, so a
     sampler that cannot fit must report the failure rather than
     silently becoming the one it is being compared with.
  5. A class excluded from the METRIC is excluded from the OBJECTIVE.
     Both the resampling strategy and the class weights are restricted
     to config.EVAL_LABELS. See the note below — this was wrong in the
     first version and it silently corrupted four of five folds.

Why rule 5 exists
-----------------
DS1 holds 8 Class Q beats in total. Per training fold that lands as
4, 6 or 8 depending on which patient the fold puts where:

    fold 0: 6    fold 1: 6    fold 2: 8    fold 3: 4    fold 4: 8

The first version guarded against this with MIN_SAMPLES_TO_RESAMPLE = 6,
which does not work: 6 and 8 are not below 6. Two things followed.

  * Resampling. Q entered the strategy in four of the five folds and
    was oversampled from 6-8 real beats to roughly 12,800 synthetic
    ones. Every adasyn row produced before this fix is void.

  * Class weights. Inverse-frequency weighting makes every class
    contribute equally to the loss, so 6 Q beats carried the same total
    gradient as 36,659 N beats — a per-sample weight 126x that of S and
    6,110x that of N. Every class_weight and focal row produced before
    this fix is void, focal included, since it takes these weights as
    its alpha.

Restricting both to EVAL_LABELS fixes the cause rather than the symptom.
MIN_SAMPLES_TO_RESAMPLE stays as a genuine neighbour guard for S, V and
F in a pathological fold.
"""
import numpy as np

import config as C

STRATEGIES = ['none', 'class_weight', 'focal', 'adasyn', 'smote']

# Minority classes are oversampled to at most this fraction of the
# majority class count. 0.35 follows the recipe reported for compact
# CNNs on MIT-BIH; full parity (1.0) over-corrects.
CAP_RATIO = 0.35

# A class needs at least this many samples in the training fold to be
# synthesised from — interpolation needs neighbours. This is NOT the
# mechanism that keeps Q out; EVAL_LABELS is. See the module docstring.
MIN_SAMPLES_TO_RESAMPLE = 6

# The two samplers spell the neighbour argument differently: ADASYN
# takes `n_neighbors`, SMOTE takes `k_neighbors`. Passing the wrong one
# raises TypeError, which silently disabled the SMOTE fallback in the
# first version of this module until it was caught.
_SAMPLERS = {'adasyn': 'n_neighbors', 'smote': 'k_neighbors'}


def class_weights(y, n_classes=5, eval_labels=None):
    """Inverse-frequency weights over the EVALUATED classes only.

    Classes outside eval_labels get weight 0, so they contribute no
    gradient. Among the evaluated classes the weights are normalised to
    mean 1, which is the same relative scheme as sklearn's
    class_weight='balanced'.
    """
    if eval_labels is None:
        eval_labels = C.EVAL_LABELS

    counts = np.bincount(y, minlength=n_classes).astype(np.float64)

    mask = np.zeros(n_classes, dtype=bool)
    mask[list(eval_labels)] = True
    mask &= counts > 0                     # a class absent from this fold

    w = np.zeros(n_classes, dtype=np.float64)
    if not mask.any():                     # degenerate; should not happen
        return np.ones(n_classes, dtype=np.float32)

    c = counts[mask]
    w[mask] = c.sum() / (mask.sum() * c)
    w[mask] /= w[mask].mean()
    return w.astype(np.float32)


def sample_weights(y, n_classes=5, eval_labels=None):
    """Per-sample weights for XGBoost / sklearn `sample_weight`.

    Normalised so the MEAN SAMPLE weight is 1. That matters: the class
    weights above are normalised across classes, which for N leaves a
    per-sample weight near 0.0008. Handed to a regularised fit as-is,
    that shrinks the data term relative to the L2 penalty and the model
    underfits. Scaling to mean 1 keeps the effective regularisation
    strength the same as an unweighted fit, which is also what
    sklearn's 'balanced' does.
    """
    w = class_weights(y, n_classes, eval_labels)[y]
    m = w.mean()
    return (w / m).astype(np.float32) if m > 0 else w.astype(np.float32)


def apply_resample(X, y, kind, cap_ratio=CAP_RATIO, seed=0, eval_labels=None):
    """Oversample minorities to `cap_ratio` x majority count.

    `kind` is 'adasyn' or 'smote'. There is deliberately NO fallback
    from one to the other: the two are separate rows in the grid and
    exist to be compared, so a sampler that cannot fit reports that in
    the returned note instead of quietly handing the work to its rival.
    A row of results produced by the other sampler would be worse than
    no row at all.

    Only classes in eval_labels are synthesised — see rule 5 in the
    module docstring.

    X must be 2-D. For the CNN the waveform and the RR columns are
    stacked into one matrix before this call and split afterwards, so
    the sampler interpolates both together — see the caveat in the
    module docstring of models.py. Interpolating between raw waveforms
    can produce beats that are not physiologically plausible, which is
    both why the cap matters and why resampling helps a model fitted on
    morphology descriptors and hurts one fitted on the waveform itself.

    Returns (X, y, note). The note records what actually happened and
    is written into each prediction file's metadata.
    """
    from imblearn.over_sampling import ADASYN, SMOTE

    if eval_labels is None:
        eval_labels = C.EVAL_LABELS

    sampler_cls = {'adasyn': ADASYN, 'smote': SMOTE}[kind]
    kw_name = _SAMPLERS[kind]

    counts = np.bincount(y, minlength=5)
    n_major = counts.max()
    target = int(round(cap_ratio * n_major))

    # Only evaluated classes below target are raised, and only to target.
    #
    # A class with a handful of members additionally cannot be
    # synthesised from at all — both samplers interpolate between a
    # sample and its nearest same-class neighbours. Leaving such a class
    # IN the strategy makes the whole fit_resample call raise, which
    # returns the data completely unresampled, so the row comes out byte
    # identical to the 'none' row and the comparison is silently void.
    # That happened once already, before MIN_SAMPLES_TO_RESAMPLE existed.
    strategy = {c: target for c in eval_labels
                if MIN_SAMPLES_TO_RESAMPLE <= counts[c] < target}
    if not strategy:
        return X, y, 'skipped_no_eligible_class'

    skipped = [c for c in eval_labels
               if 0 < counts[c] < MIN_SAMPLES_TO_RESAMPLE]

    note = '' if not skipped else \
        '+skipped' + ''.join(str(c) for c in skipped)

    k = min(5, max(1, min(counts[c] for c in strategy) - 1))

    try:
        sampler = sampler_cls(sampling_strategy=strategy,
                              random_state=seed, **{kw_name: k})
        Xr, yr = sampler.fit_resample(X, y)
        return Xr, yr, kind + note
    except Exception as exc:                          # noqa: BLE001
        return X, y, f'{kind}_failed:{type(exc).__name__}{note}'


def apply_adasyn(X, y, cap_ratio=CAP_RATIO, seed=0):
    """Back-compat wrapper. Prefer apply_resample(X, y, 'adasyn')."""
    return apply_resample(X, y, 'adasyn', cap_ratio=cap_ratio, seed=seed)


def focal_loss_torch(logits, target, gamma=2.0, alpha=None):
    """Multi-class focal loss.

    Standard cross-entropy keeps re-teaching the ~46,000 obvious normal
    beats. Focal loss down-weights examples the model already gets right
    so gradient updates concentrate on the hard, rare ones.

        FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: optional per-class tensor (defaults to uniform). When it comes
    from class_weights() above, non-evaluated classes carry weight 0 and
    so contribute nothing to the loss.
    """
    import torch
    import torch.nn.functional as F

    logp = F.log_softmax(logits, dim=1)
    logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()

    loss = -((1.0 - p_t) ** gamma) * logp_t
    if alpha is not None:
        loss = loss * alpha.to(logits.device)[target]
    return loss.mean()