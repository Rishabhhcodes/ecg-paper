"""
Boxes 8 & 11 — run the experimental grid on DS1 with patient-grouped CV.

    3 models x 5 imbalance strategies x 2 RR settings x 5 folds

DS2 is never opened by this script. Not once. That is the protocol.

Macro-F1 is averaged over config.EVAL_CLASSES (N, S, V, F). Q is still a
model output and its per-class F1 is still reported, but it is left out
of the average — see the note in config.py for why. Q is also left out
of the training objective entirely; see rule 5 in imbalance.py.

Every fold writes its validation predictions to
artifacts/preds/{model}_{strategy}_{rr}_f{fold}.npz, and the script skips
any fold whose file already exists. So a Colab disconnect costs you one
fold, not the run. Stages 3-5 read those prediction files and never
retrain.

    python src/run_grid.py                      # everything
    python src/run_grid.py --models lr xgb      # CPU-only subset
    python src/run_grid.py --models cnn         # the GPU part
    python src/run_grid.py --subset 5000        # fast smoke test
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from imbalance import STRATEGIES, apply_resample, sample_weights
from models import build_features, make_lr, make_xgb

N_FOLDS = 5
FOLDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'folds.json')


def pred_dir(seed):
    """Where this seed's per-fold predictions live.

    Seed 0 keeps the original flat path so the predictions already on disk
    stay valid; every other seed writes to a sibling directory. Without
    this the filenames would collide and a second seed would silently
    overwrite the first — the predictions carry no seed in their name.
    """
    return (os.path.join(C.OUT_DIR, 'preds') if seed == 0
            else os.path.join(C.OUT_DIR, f'preds_seed{seed}'))


def summary_path(seed):
    return (os.path.join(C.OUT_DIR, 'grid_summary.json') if seed == 0
            else os.path.join(C.OUT_DIR, f'grid_summary_seed{seed}.json'))

# Strategies that act on the training data rather than on the objective.
RESAMPLING_STRATEGIES = ('adasyn', 'smote')


def load_folds(y, pid):
    """Read the frozen fold assignment from src/folds.json.

    The folds are NOT recomputed here. StratifiedGroupKFold is deterministic
    for a given scikit-learn version but not across versions: scikit-learn
    1.6.1 (Colab) and 1.9.1 (laptop) partitioned the same 22 patients
    differently, so the LR/XGBoost rows and the CNN rows of the Stage 2 grid
    silently described different splits of DS1. Nothing errored; the two
    halves simply were not comparable.

    src/make_folds.py produced the file once. It records which fold each
    PATIENT belongs to, so patient-disjointness is visible on inspection
    rather than trusted.
    """
    if not os.path.exists(FOLDS_FILE):
        raise FileNotFoundError(
            f'{FOLDS_FILE} is missing.\n'
            '  It is committed to the repository -- pull main, or ask for a '
            'copy.\n'
            '  Do NOT run make_folds.py to recreate it: a different '
            'scikit-learn\n'
            '  version produces a different split, which is the exact '
            'problem this\n'
            '  file exists to prevent.')

    with open(FOLDS_FILE) as fh:
        spec = json.load(fh)

    assert spec['n_beats'] == len(y), (
        f"folds.json is for {spec['n_beats']:,} beats, this ds1.npz has "
        f'{len(y):,} — the dataset was rebuilt, so the folds no longer apply')

    pf = {str(k): int(v) for k, v in spec['patient_fold'].items()}
    missing = sorted({str(p) for p in np.unique(pid)} - set(pf))
    assert not missing, f'patients absent from folds.json: {missing}'

    assign = np.array([pf[str(p)] for p in pid])
    idx = np.arange(len(y))
    folds = [(idx[assign != f], idx[assign == f]) for f in range(spec['n_folds'])]

    gen = spec.get('generated_with', {})
    print(f"folds: {FOLDS_FILE} "
          f"(frozen, made with sklearn {gen.get('sklearn', '?')})")
    return folds


def compute_folds_for_subset(X, y, pid, seed):
    """Folds for a --subset smoke test only.

    A subset is not the real dataset, so the frozen assignment does not
    apply. These folds are version-dependent and the predictions they
    produce must never be kept — the notebook deletes them, and the cache
    guard in run() rejects them anyway.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    print('SMOKE TEST: folds computed on the fly, results are not usable')
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=seed)
    return list(gkf.split(X, y, groups=pid))


def tag(model, strategy, use_rr, fold):
    return f'{model}_{strategy}_{"rr" if use_rr else "norr"}_f{fold}'


def fit_sklearn_like(kind, strategy, Xtr, rrtr, ytr, Xva, rrva, use_rr, seed):
    """LR and XGBoost share a shape: build features, resample, fit, predict."""
    Ftr = build_features(Xtr, rrtr, use_rr, C.SEGMENTS, kind)
    Fva = build_features(Xva, rrva, use_rr, C.SEGMENTS, kind)

    note = 'none'
    # Resampling happens in FEATURE space, on Ftr — not on the raw
    # waveform. For XGBoost that means ADASYN and SMOTE interpolate
    # between morphology descriptors, where a midpoint is still a valid
    # statistic. For LR, whose features ARE the raw waveform, they
    # interpolate waveforms, exactly as in the CNN path below.
    #
    # Fitted on the training fold only — never the validation fold.
    if strategy in RESAMPLING_STRATEGIES:
        Ftr, ytr, note = apply_resample(Ftr, ytr, strategy, seed=seed)

    # Both families take class weighting the same way, through
    # sample_weight. imbalance.sample_weights() restricts the weights to
    # config.EVAL_LABELS, so Q carries no gradient. LogisticRegression's
    # own class_weight='balanced' would include Q and is not used.
    sw = sample_weights(ytr) if strategy == 'class_weight' else None

    if kind == 'lr':
        model = make_lr(strategy, seed)
        fit_kw = {} if sw is None else {'logisticregression__sample_weight': sw}
        model.fit(Ftr, ytr, **fit_kw)
    else:
        model = make_xgb(seed)
        model.fit(Ftr, ytr, sample_weight=sw, verbose=False)

    return model.predict_proba(Fva), {'resample': note, 'n_train': int(len(ytr))}


def run(models, strategies, rr_settings, subset=None, seed=0, epochs=40):
    PRED_DIR = pred_dir(seed)
    os.makedirs(PRED_DIR, exist_ok=True)
    if seed != 0:
        print(f'seed {seed}: writing to {PRED_DIR}')

    d = np.load(os.path.join(C.OUT_DIR, 'ds1.npz'))
    X, rr, y, pid = d['X'], d['rr'], d['y'], d['pid']

    if subset:                                   # smoke test on a subsample
        idx = np.random.default_rng(seed).choice(len(y), min(subset, len(y)),
                                                 replace=False)
        X, rr, y, pid = X[idx], rr[idx], y[idx], pid[idx]

    print(f'DS1: {len(y):,} beats from {len(np.unique(pid))} patients')
    print(f'grid: {len(models)} models x {len(strategies)} strategies x '
          f'{len(rr_settings)} rr x {N_FOLDS} folds\n')

    # Box 8 — folds split by patient, never by beat. A plain KFold here
    # would silently re-create the leak Stage 1 exists to remove.
    #
    # The assignment was made with StratifiedGroupKFold rather than plain
    # GroupKFold: both keep every patient inside one fold, but the
    # stratified version also balances class proportions across folds. On
    # DS1 that matters for V — plain GroupKFold produced a fold holding
    # just 2 V beats out of 3,788, and an F1 computed on 2 samples is
    # noise that drags the pooled V score down.
    #
    # It cannot help S or F. Record 209 holds 383 of DS1's 943 S beats,
    # and record 208 holds 372 of its 414 F beats; a class concentrated
    # in one patient cannot be spread across folds by any scheme that
    # keeps patients intact. The cross-validated F score in particular is
    # not interpretable, which is a property of the database rather than
    # of this pipeline, and is worth stating in the paper.
    #
    # That assignment is now READ FROM src/folds.json rather than
    # recomputed — see load_folds().
    folds = (compute_folds_for_subset(X, y, pid, seed) if subset
             else load_folds(y, pid))

    for f, (tr, va) in enumerate(folds):
        overlap = set(pid[tr]) & set(pid[va])
        assert not overlap, f'fold {f}: patient overlap {overlap}'

    summary = []
    for model in models:
        for strategy in strategies:
            if model in ('lr', 'xgb') and strategy == 'focal':
                print(f'{model:4} focal        -- skipped '
                      f'(focal loss is a neural-net objective)')
                continue

            for use_rr in rr_settings:
                f1s, t0 = [], time.time()
                label = f'{model:4} {strategy:12} {"rr " if use_rr else "norr"}'

                for f, (tr, va) in enumerate(folds):
                    path = os.path.join(PRED_DIR,
                                        tag(model, strategy, use_rr, f) + '.npz')
                    if os.path.exists(path):                # resume support
                        cached = np.load(path)
                        # Guard: a cached fold from a different data
                        # configuration (e.g. a --subset smoke test) must
                        # never be silently reused as a real result.
                        if (len(cached['y_true']) == len(va)
                                and np.array_equal(cached['val_idx'], va)):
                            f1s.append(float(cached['macro_f1']))
                            continue
                        print(f'      fold {f}: cached prediction does not '
                              f'match current split — recomputing')

                    if model == 'cnn':
                        from train_cnn import train_cnn
                        Xtr, rrtr, ytr = X[tr], rr[tr], y[tr]
                        note = 'none'
                        # The CNN resamples in WAVEFORM space. The 216
                        # waveform columns and the 4 RR columns are stacked
                        # into one matrix, so the sampler interpolates both
                        # together and the shape-to-timing correspondence is
                        # not preserved in synthetic beats. This is the
                        # counterpart to the feature-space path above.
                        if strategy in RESAMPLING_STRATEGIES:
                            flat = np.hstack([Xtr, rrtr])
                            flat, ytr, note = apply_resample(
                                flat, ytr, strategy, seed=seed)
                            Xtr = flat[:, :X.shape[1]]
                            rrtr = flat[:, X.shape[1]:]
                        proba, meta = train_cnn(
                            Xtr, rrtr, ytr, X[va], rr[va], y[va],
                            use_rr=use_rr, strategy=strategy,
                            epochs=epochs, seed=seed, verbose=False)
                        meta['resample'] = note
                    else:
                        proba, meta = fit_sklearn_like(
                            model, strategy, X[tr], rr[tr], y[tr],
                            X[va], rr[va], use_rr, seed)

                    # Macro-F1 over EVAL_CLASSES only (Q excluded — see
                    # config.EVAL_CLASSES). `labels=` restricts which
                    # classes are averaged; the model still predicts all 5.
                    mf1 = f1_score(y[va], proba.argmax(1), average='macro',
                                   labels=C.EVAL_LABELS, zero_division=0)
                    f1s.append(mf1)

                    np.savez_compressed(
                        path, proba=proba.astype(np.float32),
                        y_true=y[va], pid=pid[va], val_idx=va,
                        macro_f1=mf1, meta=json.dumps(meta))

                # Per-class F1 across the pooled folds. `labels=` is passed
                # explicitly so the array is always length 5 and its indices
                # line up with CLASS_TO_INT even when a class (Q) is absent
                # from a fold entirely. Q's value is reported for honesty but
                # is not part of the macro average above.
                s_f1 = f1_score(
                    np.concatenate([y[va] for _, va in folds]),
                    np.concatenate([
                        np.load(os.path.join(
                            PRED_DIR, tag(model, strategy, use_rr, f) + '.npz')
                        )['proba'].argmax(1) for f in range(N_FOLDS)]),
                    average=None, labels=list(range(len(C.CLASS_NAMES))),
                    zero_division=0)

                print(f'{label}  macroF1 {np.mean(f1s):.4f} '
                      f'+/- {np.std(f1s):.4f}   '
                      f'S={s_f1[C.CLASS_TO_INT["S"]]:.3f} '
                      f'V={s_f1[C.CLASS_TO_INT["V"]]:.3f} '
                      f'F={s_f1[C.CLASS_TO_INT["F"]]:.3f}   '
                      f'[{time.time() - t0:.0f}s]')

                summary.append({
                    'model': model, 'strategy': strategy, 'use_rr': bool(use_rr),
                    'seed': int(seed),
                    'eval_classes': C.EVAL_CLASSES,
                    'macro_f1_mean': float(np.mean(f1s)),
                    'macro_f1_std': float(np.std(f1s)),
                    'per_class_f1': {c: float(s_f1[i])
                                     for i, c in enumerate(C.CLASS_NAMES)},
                })

    out = summary_path(seed)
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\nWrote {len(summary)} configurations to {out}')
    print(f'Per-fold predictions in {PRED_DIR}/ — Stages 3-5 read these.')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['lr', 'xgb', 'cnn'])
    p.add_argument('--strategies', nargs='+', default=STRATEGIES)
    p.add_argument('--rr', nargs='+', type=int, default=[1, 0],
                   help='1 = with RR features, 0 = without (ablation)')
    p.add_argument('--subset', type=int, default=None)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    run(a.models, a.strategies, [bool(r) for r in a.rr],
        subset=a.subset, seed=a.seed, epochs=a.epochs)