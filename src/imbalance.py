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
"""
import numpy as np

STRATEGIES = ['none', 'class_weight', 'focal', 'adasyn', 'smote']

# Minority classes are oversampled to at most this fraction of the
# majority class count. 0.35 follows the recipe reported for compact
# CNNs on MIT-BIH; full parity (1.0) over-corrects.
CAP_RATIO = 0.35

# A class needs at least this many samples in the training fold to be
# synthesised from — interpolation needs neighbours. Below it, the class
# is excluded from the resampling strategy rather than being allowed to
# fail the whole call. See the comment in apply_resample.
MIN_SAMPLES_TO_RESAMPLE = 6

# The two samplers spell the neighbour argument differently: ADASYN
# takes `n_neighbors`, SMOTE takes `k_neighbors`. Passing the wrong one
# raises TypeError, which silently disabled the SMOTE fallback in the
# first version of this module until it was caught.
_SAMPLERS = {'adasyn': 'n_neighbors', 'smote': 'k_neighbors'}


def class_weights(y, n_classes=5):
    """Inverse-frequency weights, normalised to mean 1."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)


def sample_weights(y, n_classes=5):
    """Per-sample weights for XGBoost / sklearn `sample_weight`."""
    return class_weights(y, n_classes)[y]


def apply_resample(X, y, kind, cap_ratio=CAP_RATIO, seed=0):
    """Oversample minorities to `cap_ratio` x majority count.

    `kind` is 'adasyn' or 'smote'. There is deliberately NO fallback
    from one to the other: the two are separate rows in the grid and
    exist to be compared, so a sampler that cannot fit reports that in
    the returned note instead of quietly handing the work to its rival.
    A row of results produced by the other sampler would be worse than
    no row at all.

    X must be 2-D. For the CNN the waveform and the RR columns are
    stacked into one matrix before this call and split afterwards, so
    the sampler interpolates both together — see the caveat in the
    module docstring of models.py. Interpolating between raw waveforms
    can produce beats that are not physiologically plausible, which is
    both why the cap matters and why ADASYN is the best strategy for
    XGBoost and the worst for the CNN.

    Returns (X, y, note). The note records what actually happened and
    is written into each prediction file's metadata.
    """
    from imblearn.over_sampling import ADASYN, SMOTE

    sampler_cls = {'adasyn': ADASYN, 'smote': SMOTE}[kind]
    kw_name = _SAMPLERS[kind]

    counts = np.bincount(y, minlength=5)
    n_major = counts.max()
    target = int(round(cap_ratio * n_major))

    # Only classes currently below target are raised, and only to target.
    #
    # Classes with too few samples are left out of the strategy entirely.
    # ADASYN and SMOTE both interpolate between a sample and its nearest
    # neighbours of the same class, so a class with a handful of members
    # cannot be synthesised from. Q is the usual culprit: 8 beats in DS1,
    # which lands as 0-3 per fold.
    #
    # Leaving such a class IN the strategy makes the whole fit_resample
    # call raise, which returns the data completely unresampled — so the
    # 'adasyn' row comes out byte identical to the 'none' row and the
    # comparison is silently void. That is exactly what happened before
    # MIN_SAMPLES_TO_RESAMPLE existed. Excluding it lets S, V and F still
    # be resampled, which is the point.
    strategy = {c: target for c in range(5)
                if MIN_SAMPLES_TO_RESAMPLE <= counts[c] < target}
    if not strategy:
        return X, y, 'skipped_no_eligible_class'

    skipped = [c for c in range(5)
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

    alpha: optional per-class tensor (defaults to uniform).
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