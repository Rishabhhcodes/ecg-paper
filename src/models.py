"""
Box 9 — the three model families.

  9a  Logistic Regression   carried over from Paper #1, for comparability
  9b  XGBoost               morphology descriptors + RR features
  9c  1D-CNN                waveform, with RR features fused before the head

Every model takes an `use_rr` flag. Running the whole grid twice, with
and without RR, is the ablation (box 17) that isolates how much of the
Class S gain comes from timing rather than architecture.
"""
import numpy as np


# ----------------------------------------------------------------------
# Feature construction
# ----------------------------------------------------------------------
def morphology_features(X, segments):
    """Compact descriptors for tree models.

    Trees treat each input column independently, so handing XGBoost 216
    raw timepoints wastes it — it cannot see shape. These descriptors
    give it the shape information explicitly.

    This is also the reason ADASYN and SMOTE behave differently for the
    two model families: here they interpolate between statistics, where
    a midpoint is still a valid statistic. On the raw waveform, which is
    what LR and the CNN receive, a midpoint between two beats whose
    R-peaks sit at different sample offsets is a smeared or doubled QRS
    that no heart produces.
    """
    feats = [
        X.min(1), X.max(1), X.mean(1), X.std(1),
        np.ptp(X, axis=1),
        (X ** 2).sum(1),                             # energy
        np.abs(np.diff(X, axis=1)).mean(1),          # mean |slope|
        X.argmax(1).astype(np.float32),              # R-peak drift
    ]
    for name, (a, b) in segments.items():
        seg = X[:, a:b]
        feats += [seg.mean(1), seg.std(1), seg.min(1), seg.max(1),
                  (seg ** 2).sum(1)]
    # coarse shape: 36-point downsample of the window
    feats.append(X[:, ::6].T)
    out = [f if f.ndim == 2 else f[None, :] for f in feats]
    return np.vstack(out).T.astype(np.float32)


def build_features(X, rr, use_rr, segments, kind):
    """Assemble the input matrix for a given model family."""
    if kind == 'lr':
        base = X                                     # mirrors Paper #1
    elif kind == 'xgb':
        base = morphology_features(X, segments)
    else:
        raise ValueError(kind)
    return np.hstack([base, rr]).astype(np.float32) if use_rr else base.astype(np.float32)


# ----------------------------------------------------------------------
# 9a — Logistic Regression
# ----------------------------------------------------------------------
def make_lr(strategy, seed=0):
    """Logistic regression, standardised then fitted.

    The scaler matters. Per-record z-scoring (Stage 1) centres each
    patient's signal, but it leaves the columns on very different scales:
    the QRS peak sits many standard deviations above baseline, while the
    RR features are seconds (~0.8) and ratios (~1.0). lbfgs struggles on
    that and stops at the iteration cap without converging.

    A non-converged baseline is a baseline that was not given a fair
    shot, which would make "the CNN beats LR" a weaker claim than it
    looks and would confound the inter-patient drop in C1 with an
    optimisation artefact.

    Pipeline, not a pre-scaled array: scikit-learn fits the scaler on the
    training fold only when the pipeline is fitted, so the validation
    fold's statistics never leak backwards.

    Class weighting is NOT set here. It used to be class_weight='balanced',
    which weights all five classes including Q — and with 6-8 Q beats in
    a training fold that hands Q a per-sample weight 126x that of S. The
    caller now passes sample_weight built by imbalance.sample_weights(),
    which is restricted to the evaluated classes. `strategy` is kept in
    the signature so the call sites stay uniform.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )


# ----------------------------------------------------------------------
# 9b — XGBoost
# ----------------------------------------------------------------------
def make_xgb(seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', num_class=5,
        tree_method='hist', n_jobs=-1, random_state=seed,
        eval_metric='mlogloss',
    )


# ----------------------------------------------------------------------
# 9c — 1D-CNN with RR fusion
# ----------------------------------------------------------------------
def make_cnn(use_rr, n_classes=5, seed=0):
    """Compact 4-block CNN. ~150k parameters.

    Deliberately small: the literature shows compact CNNs match far
    larger models on this task, and Stage 5 reports parameter count and
    latency as an embedded-feasibility claim.

    RR features bypass the convolutional stack and are concatenated at
    the head. Convolution over a single beat window physically cannot
    recover inter-beat timing, so the features must enter here.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [1, 32, 64, 128, 128]
            blocks = []
            for i in range(4):
                blocks += [
                    nn.Conv1d(ch[i], ch[i + 1], kernel_size=5, padding=2),
                    nn.BatchNorm1d(ch[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                ]
            self.conv = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.3)

            # RR branch. Three things matter here, and the first version of
            # this model got all three wrong, which left the CNN scoring
            # S-class F1 around 0.06 while XGBoost on the same folds reached
            # 0.27. The CNN was reading morphology fine (it beat XGBoost on V)
            # and simply failing to use timing.
            #
            #   1. The four RR values are raw: seconds (~0.8) and ratios
            #      (~1.0). The conv features arrive from BatchNorm -> ReLU ->
            #      global average pooling on an entirely different scale.
            #      BatchNorm1d here puts both branches on comparable footing.
            #
            #   2. Four inputs beside 128 conv features is 3% of the vector.
            #      Projecting them to RR_DIM gives the timing signal enough
            #      width for the head to actually lean on it.
            #
            #   3. Dropout used to be applied AFTER the concatenation, so
            #      each RR feature was zeroed 30% of the time. With only four
            #      features carrying the sole timing signal, that is not
            #      regularisation, it is deletion. Dropout now applies to the
            #      conv branch only.
            RR_DIM = 32
            self.use_rr = use_rr
            if use_rr:
                self.rr_proj = nn.Sequential(
                    nn.BatchNorm1d(4),
                    nn.Linear(4, RR_DIM), nn.ReLU(inplace=True),
                    nn.Linear(RR_DIM, RR_DIM), nn.ReLU(inplace=True),
                )

            head_in = ch[-1] + (RR_DIM if use_rr else 0)
            self.head = nn.Sequential(
                nn.Linear(head_in, 64), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(64, n_classes),
            )

        def forward(self, x, rr=None):
            h = self.pool(self.conv(x.unsqueeze(1))).squeeze(-1)
            h = self.drop(h)                      # conv branch only
            if self.use_rr:
                h = torch.cat([h, self.rr_proj(rr)], dim=1)
            return self.head(h)

    return CNN1D()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)