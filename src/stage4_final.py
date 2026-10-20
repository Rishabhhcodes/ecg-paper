"""
Stage 4 — the single evaluation on DS2.

    python src/stage4_final.py --confirm

THIS IS THE ONE-SHOT STAGE. DS2 has been sealed since Stage 1. Every
hyperparameter, architecture choice, loss function and threshold in this
project was selected without it. Opening it is what turns those choices
into an honest estimate of generalisation — and it only works once,
because a number you are allowed to improve is not a test score.

The script enforces that rather than trusting it. The first successful
run writes artifacts/stage4/RECEIPT.json. Every later run is refused
unless it carries --reason "...", which is appended to the receipt. The
receipt is the audit trail: if the paper ever reports a second DS2 number,
the receipt says why there was one.

    python src/stage4_final.py --confirm
    python src/stage4_final.py --confirm --reason "reviewer asked for XGBoost"

What it does
------------
1. Trains each locked configuration on DS1 and predicts DS2 once.
2. Reports AAMI metrics: per-class sensitivity and positive predictivity,
   macro-F1, and the full confusion matrix.
3. If Stage 2 predictions are present, additionally reports whether the
   DS1-fitted temperature transfers to unseen patients.

The configurations are hardcoded below. That is deliberate: a --model
flag would let someone run DS2 five times with different options and
report the best, which is the exact failure this stage exists to avoid.

Early stopping, and why it does not leak
----------------------------------------
train_cnn needs a validation set to decide when to stop. Handing it DS2
would select the stopping epoch using the test set. Instead one DS1 fold
is reserved for early stopping and the model trains on the other four,
so DS2 is never consulted during fitting.

Each seed uses a DIFFERENT fold for this (fold = seed % 5), so across
five seeds all five folds are used exactly once and the reported spread
covers both the initialisation and the choice of stopping set.

A side effect worth stating in the paper: the training set is then the
same size as every Stage 2 model's, so the DS1 cross-validated score and
the DS2 score are directly comparable, and the gap between them is purely
the inter-patient generalisation gap rather than a difference in how much
data each model saw.
"""
import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from imbalance import apply_resample, sample_weights
from models import build_features, make_lr, make_xgb

N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
PROB_FLOOR = 1e-12
ECE_BINS = 15
RESAMPLING = ('adasyn', 'smote')

OUT = os.path.join(C.OUT_DIR, 'stage4')
RECEIPT = os.path.join(OUT, 'RECEIPT.json')

# Locked on the Stage 2 evidence, before DS2 was opened.
#
#   cnn / class_weight and cnn / focal are reported together because
#   across five seeds they cannot be separated on either macro-F1 or
#   Class S. Picking one would be an arbitrary choice presented as a
#   result; reporting both is the honest form.
#
#   xgb / class_weight stands in for a three-way tie with SMOTE and
#   ADASYN (within 0.008, and single-seed, so not rankable at all).
#
#   lr / none is the best LR configuration — every correction cost it
#   more than it returned.
#
# RR features are on everywhere. That is the paper's finding, not a
# tunable: 26 of 26 comparisons favoured them.
CONFIGS = [
    ('cnn', 'class_weight', True),
    ('cnn', 'focal', True),
    ('xgb', 'class_weight', True),
    ('lr', 'none', True),
]


# ------------------------------------------------------------ the lock

def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def check_lock(reason):
    """Refuse a second opening of DS2 unless a reason is recorded."""
    if not os.path.exists(RECEIPT):
        return []
    with open(RECEIPT) as fh:
        prior = json.load(fh)
    runs = prior if isinstance(prior, list) else [prior]
    if not reason:
        first = runs[0].get('finished_at', '?')
        raise SystemExit(
            '\n'.join([
                'REFUSED — DS2 has already been opened.',
                '',
                f'  first opened : {first}',
                f'  openings     : {len(runs)}',
                f'  receipt      : {RECEIPT}',
                '',
                'A test set re-run until the number improves is not a test',
                'set. If there is a legitimate reason to open it again --',
                'a bug in this script, a reviewer request -- state it:',
                '',
                '    python src/stage4_final.py --confirm --reason "..."',
                '',
                'The reason is appended to the receipt and belongs in the',
                'paper if the second number is the one reported.',
            ]))
    print(f'*** DS2 has been opened {len(runs)} time(s) before.')
    print(f'*** Reason given for opening it again: {reason}\n')
    return runs


def write_receipt(prior, results, reason, elapsed):
    entry = {
        'finished_at': datetime.datetime.now().astimezone().isoformat(),
        'git_commit': git_commit(),
        'reason': reason or 'first opening',
        'configs': [{'model': m, 'strategy': s, 'use_rr': r}
                    for m, s, r in CONFIGS],
        'seeds': SEEDS,
        'ds1_sha_shape': results['_inputs']['ds1'],
        'ds2_sha_shape': results['_inputs']['ds2'],
        'elapsed_min': round(elapsed / 60, 1),
        'headline': results['_headline'],
    }
    with open(RECEIPT, 'w') as fh:
        json.dump(prior + [entry], fh, indent=2)


# --------------------------------------------------------------- metrics

def aami(y_true, y_pred):
    """Per-class sensitivity and positive predictivity, the convention in
    the inter-patient literature, plus F1 for continuity with Stage 2."""
    n = len(C.CLASS_NAMES)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    out = {}
    for c in C.EVAL_CLASSES:
        i = C.CLASS_TO_INT[c]
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        se = tp / (tp + fn) if tp + fn else 0.0
        pp = tp / (tp + fp) if tp + fp else 0.0
        out[c] = {'n': tp + fn, 'TP': tp, 'FN': fn, 'FP': fp,
                  'Se': se, 'PP': pp,
                  'F1': 2 * se * pp / (se + pp) if se + pp else 0.0}
    return cm, out


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average='macro',
                          labels=C.EVAL_LABELS, zero_division=0))


def ece(conf, correct, bins=ECE_BINS):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


# ---------------------------------------------------- temperature from DS1

def fit_T_from_stage2(model, strategy, use_rr, seed):
    """Fit one temperature on the Stage 2 out-of-fold DS1 predictions.

    DS2 is not involved and cannot be: these files were written before it
    existed to this project. Returns None if the files are absent, in
    which case the calibration-transfer result is simply not reported.

    A caveat for the paper: those predictions come from models trained on
    four folds, while the Stage 4 model is trained on four folds too (one
    is held out for early stopping), so the two are comparable -- but they
    are not the same model, and a temperature is not guaranteed to carry
    across. That is exactly what this measures.
    """
    d = (os.path.join(C.OUT_DIR, 'preds') if seed == 0
         else os.path.join(C.OUT_DIR, f'preds_seed{seed}'))
    rr = 'rr' if use_rr else 'norr'
    P, Y = [], []
    for f in range(N_FOLDS):
        p = os.path.join(d, f'{model}_{strategy}_{rr}_f{f}.npz')
        if not os.path.exists(p):
            return None
        z = np.load(p)
        P.append(z['proba'].astype(np.float64))
        Y.append(z['y_true'])
    z = np.log(np.clip(np.vstack(P), PROB_FLOOR, None))
    y = np.concatenate(Y)
    keep = np.isin(y, C.EVAL_LABELS)
    z, y = z[keep], y[keep]

    def nll(T):
        zt = z / T
        return float(-(zt - logsumexp(zt, axis=1, keepdims=True))[
            np.arange(len(y)), y].mean())

    return float(minimize_scalar(nll, bounds=(0.05, 10.0),
                                 method='bounded').x)


def temper(proba, T):
    z = np.log(np.clip(proba, PROB_FLOOR, None)) / T
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


# -------------------------------------------------------------- training

def train_and_predict(model, strategy, use_rr, seed, d1, d2):
    """Fit on DS1 and predict DS2 once. DS2 is read, never fitted on."""
    X1, rr1, y1, pid1 = d1
    X2, rr2 = d2[0], d2[1]

    if model == 'cnn':
        # One DS1 fold is reserved for early stopping so that DS2 is
        # never consulted while fitting. Which fold rotates with the seed.
        from train_cnn import train_cnn
        folds = load_folds(y1, pid1)
        tr, es = folds[seed % N_FOLDS]
        Xtr, rrtr, ytr = X1[tr], rr1[tr], y1[tr]
        note = 'none'
        if strategy in RESAMPLING:
            flat = np.hstack([Xtr, rrtr])
            flat, ytr, note = apply_resample(flat, ytr, strategy, seed=seed)
            Xtr, rrtr = flat[:, :X1.shape[1]], flat[:, X1.shape[1]:]
        # train_cnn early-stops on the DS1 fold passed as its validation
        # set and, via predict_extra, scores DS2 once with the restored
        # best-epoch weights. DS2 never touches the stopping decision.
        _, meta = train_cnn(Xtr, rrtr, ytr, X1[es], rr1[es], y1[es],
                            use_rr=use_rr, strategy=strategy,
                            epochs=60, seed=seed, verbose=False,
                            predict_extra=(X2, rr2))
        proba = meta.pop('proba_extra')
        meta['resample'] = note
        meta['early_stop_fold'] = int(seed % N_FOLDS)
        meta['n_train'] = int(len(ytr))
        return proba, meta

    F1 = build_features(X1, rr1, use_rr, C.SEGMENTS, model)
    F2 = build_features(X2, rr2, use_rr, C.SEGMENTS, model)
    ytr, note = y1, 'none'
    if strategy in RESAMPLING:
        F1, ytr, note = apply_resample(F1, ytr, strategy, seed=seed)
    sw = sample_weights(ytr) if strategy == 'class_weight' else None
    if model == 'lr':
        m = make_lr(strategy, seed)
        m.fit(F1, ytr, **({} if sw is None
                          else {'logisticregression__sample_weight': sw}))
    else:
        m = make_xgb(seed)
        m.fit(F1, ytr, sample_weight=sw, verbose=False)
    return m.predict_proba(F2), {'resample': note, 'n_train': int(len(ytr))}


def load_folds(y, pid):
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folds.json')
    with open(f) as fh:
        spec = json.load(fh)
    assert spec['n_beats'] == len(y), 'folds.json does not match this ds1.npz'
    pf = {str(k): int(v) for k, v in spec['patient_fold'].items()}
    a = np.array([pf[str(p)] for p in pid])
    idx = np.arange(len(y))
    return [(idx[a != k], idx[a == k]) for k in range(spec['n_folds'])]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--confirm', action='store_true',
                    help='required — opening DS2 is not a default action')
    ap.add_argument('--reason', default=None,
                    help='required only to open DS2 again after the first time')
    a = ap.parse_args()

    if not a.confirm:
        raise SystemExit(
            'This stage opens DS2, the held-out test set, and it is meant to\n'
            'happen once. Re-read docs/stage4.md, then run:\n\n'
            '    python src/stage4_final.py --confirm\n')

    os.makedirs(OUT, exist_ok=True)
    prior = check_lock(a.reason)
    t0 = time.time()

    p1 = os.path.join(C.OUT_DIR, 'ds1.npz')
    p2 = os.path.join(C.OUT_DIR, 'ds2.npz')
    for p in (p1, p2):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p} — run build_dataset.py first')
    a1, a2 = np.load(p1), np.load(p2)
    d1 = (a1['X'], a1['rr'], a1['y'], a1['pid'])
    d2 = (a2['X'], a2['rr'], a2['y'], a2['pid'])
    y2 = d2[2]

    print('=' * 74)
    print('STAGE 4 — THE SINGLE EVALUATION ON DS2')
    print('=' * 74)
    print(f'DS1  {len(d1[2]):,} beats / {len(np.unique(d1[3]))} patients')
    print(f'DS2  {len(y2):,} beats / {len(np.unique(d2[3]))} patients')
    print('DS2 class counts: ' + '  '.join(
        f'{c} {int((y2 == C.CLASS_TO_INT[c]).sum()):,}'
        for c in C.CLASS_NAMES))
    print(f'commit {git_commit()[:10]}\n')

    results, headline = {}, {}
    for model, strategy, use_rr in CONFIGS:
        key = f'{model}_{strategy}_{"rr" if use_rr else "norr"}'
        seeds = SEEDS if model == 'cnn' else [0]
        runs = []
        print('-' * 74)
        print(f'{key}   ({len(seeds)} run(s))')
        print('-' * 74, flush=True)

        for seed in seeds:
            t = time.time()
            proba, meta = train_and_predict(model, strategy, use_rr,
                                            seed, d1, d2)
            pred = proba.argmax(1)
            cm, per = aami(y2, pred)
            mf1 = macro_f1(y2, pred)

            cal = {}
            T = fit_T_from_stage2(model, strategy, use_rr, seed)
            if T is not None:
                keep = np.isin(y2, C.EVAL_LABELS)
                pb, yy = proba[keep], y2[keep]
                pc = temper(pb, T)
                assert (pb.argmax(1) == pc.argmax(1)).all(), \
                    'temperature changed a prediction; it cannot'
                cal = {'T': T,
                       'ece_before': ece(pb.max(1), pb.argmax(1) == yy),
                       'ece_after': ece(pc.max(1), pc.argmax(1) == yy)}

            runs.append({'seed': seed, 'macro_f1': mf1, 'per_class': per,
                         'confusion': cm.tolist(), 'meta': meta,
                         'calibration': cal})
            print(f'  seed {seed}  macro-F1 {mf1:.4f}   '
                  + '  '.join(f'{c} Se {per[c]["Se"]:.3f}/+P {per[c]["PP"]:.3f}'
                              for c in ('S', 'V'))
                  + f'   [{time.time() - t:.0f}s]', flush=True)

        mf = [r['macro_f1'] for r in runs]
        spread = (max(mf) - min(mf)) / 2 if len(mf) > 1 else 0.0
        headline[key] = {'macro_f1_mean': float(np.mean(mf)),
                         'macro_f1_halfrange': float(spread),
                         'n_runs': len(runs)}
        for c in C.EVAL_CLASSES:
            for k in ('Se', 'PP', 'F1'):
                headline[key][f'{c}_{k}'] = float(
                    np.mean([r['per_class'][c][k] for r in runs]))
        results[key] = runs
        print(f'  mean macro-F1 {np.mean(mf):.4f}'
              + (f' ± {spread:.4f}' if spread else '') + '\n')

    # ------------------------------------------------------------ report
    print('=' * 74)
    print('DS2 RESULTS — AAMI per-class sensitivity and positive predictivity')
    print('=' * 74)
    print(f"{'configuration':26} {'macro-F1':>10}  "
          + '  '.join(f'{c:>14}' for c in C.EVAL_CLASSES))
    print(f"{'':26} {'':>10}  " + '  '.join(f'{"Se / +P":>14}'
                                            for _ in C.EVAL_CLASSES))
    print('-' * 74)
    for key, h in headline.items():
        sp = f" ±{h['macro_f1_halfrange']:.4f}" if h['n_runs'] > 1 else ''
        print(f"{key:26} {h['macro_f1_mean']:.4f}{sp:>8}  "
              + '  '.join(f"{h[c + '_Se']:.3f}/{h[c + '_PP']:.3f}".rjust(14)
                          for c in C.EVAL_CLASSES))

    best = max(headline, key=lambda k: headline[k]['macro_f1_mean'])
    print(f'\nConfusion matrix — {best}, first run')
    cm = np.array(results[best][0]['confusion'])
    print('            ' + ''.join(f'{c:>9}' for c in C.CLASS_NAMES)
          + '   (columns = predicted)')
    for i, c in enumerate(C.CLASS_NAMES):
        print(f'  true {c:3} ' + ''.join(f'{v:>9,}' for v in cm[i]))

    cals = [r['calibration'] for runs in results.values() for r in runs
            if r['calibration']]
    print()
    if cals:
        print('CALIBRATION TRANSFER — a DS1-fitted temperature applied to DS2')
        print(f"  T (from DS1)   {np.mean([c['T'] for c in cals]):.3f}")
        print(f"  DS2 ECE        {np.mean([c['ece_before'] for c in cals]):.4f}"
              f" -> {np.mean([c['ece_after'] for c in cals]):.4f}")
        print('  No DS2 data was used to fit T, and it cannot change an '
              'argmax,\n  so no number above moves.')
    else:
        print('Calibration transfer not reported: the Stage 2 prediction '
              'files\nwere not found, so no DS1-only temperature could be '
              'fitted.')

    elapsed = time.time() - t0
    payload = {'_inputs': {'ds1': [int(x) for x in d1[0].shape],
                           'ds2': [int(x) for x in d2[0].shape]},
               '_headline': headline, 'runs': results}
    with open(os.path.join(OUT, 'stage4_results.json'), 'w') as fh:
        json.dump(payload, fh, indent=2)
    write_receipt(prior, payload, a.reason, elapsed)

    print(f'\nWrote stage4_results.json and RECEIPT.json -> {OUT}')
    print(f'Elapsed {elapsed / 60:.1f} min.')
    print('\nDS2 is now open. There is no second first look — any further')
    print('run is recorded in the receipt and belongs in the paper.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())