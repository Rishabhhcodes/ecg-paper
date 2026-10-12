"""
Rebuild every grid summary from the prediction files.

    python src/rebuild_summary.py            # show what would change
    python src/rebuild_summary.py --write    # write the summaries

Why this exists
---------------
run_grid.py builds its summary list from scratch on every run and writes
the whole file at the end:

    summary = []
    for model in models: ...
    json.dump(summary, ...)

So a partial run REPLACES the summary rather than updating it. Running
`--models lr xgb` after a full grid leaves a 16-row grid_summary.json
where there was a 26-row one, and the ten CNN rows are gone. Nothing
errors. The per-fold predictions are untouched, because those are
written one file per fold -- only the summary is lost.

That is how a CNN configuration came to be averaged over four seeds
while the report said five: seed 0's CNN rows were missing from
grid_summary.json, and seed_report.py quietly averaged whatever it
found.

The predictions are the real result. This script treats the summaries
as what they are -- a derived view -- and regenerates them from the
.npz files, so a clobbered summary is a two-second fix rather than a
re-run. It never trains anything and cannot: it only reads .npz files.

It reproduces run_grid.py's reductions exactly:
  macro_f1_mean / macro_f1_std   mean and sd ACROSS THE FIVE FOLDS
  per_class_f1                   F1 on the POOLED beats
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

N_FOLDS = 5
MODEL_ORDER = ['lr', 'xgb', 'cnn']
STRATEGY_ORDER = ['none', 'class_weight', 'focal', 'adasyn', 'smote']
# {model}_{strategy}_{rr|norr}_f{fold}.npz, where strategy may itself
# contain an underscore ('class_weight') but the model name never does.
NAME = re.compile(r'^(?P<model>[^_]+)_(?P<strategy>.+)_(?P<rr>rr|norr)'
                  r'_f(?P<fold>\d+)\.npz$')


def seed_dirs():
    out = {}
    base = os.path.join(C.OUT_DIR, 'preds')
    if os.path.isdir(base):
        out[0] = base
    for p in sorted(glob.glob(os.path.join(C.OUT_DIR, 'preds_seed*'))):
        try:
            out[int(os.path.basename(p).replace('preds_seed', ''))] = p
        except ValueError:
            pass
    return out


def discover(directory):
    """{(model, strategy, use_rr): {fold: path}} for one seed directory."""
    combos = {}
    for p in sorted(glob.glob(os.path.join(directory, '*.npz'))):
        m = NAME.match(os.path.basename(p))
        if not m:
            continue
        key = (m['model'], m['strategy'], m['rr'] == 'rr')
        combos.setdefault(key, {})[int(m['fold'])] = p
    return combos


def summarise(key, folds, seed):
    model, strategy, use_rr = key
    per_fold, Y, P = [], [], []
    for f in range(N_FOLDS):
        d = np.load(folds[f])
        y, pred = d['y_true'], d['proba'].argmax(1)
        per_fold.append(f1_score(y, pred, average='macro',
                                 labels=C.EVAL_LABELS, zero_division=0))
        Y.append(y)
        P.append(pred)
    pooled = f1_score(np.concatenate(Y), np.concatenate(P), average=None,
                      labels=list(range(len(C.CLASS_NAMES))), zero_division=0)
    return {
        'model': model, 'strategy': strategy, 'use_rr': bool(use_rr),
        'seed': int(seed),
        'eval_classes': C.EVAL_CLASSES,
        'macro_f1_mean': float(np.mean(per_fold)),
        'macro_f1_std': float(np.std(per_fold)),
        'per_class_f1': {c: float(pooled[i])
                         for i, c in enumerate(C.CLASS_NAMES)},
    }


def order(row):
    m = MODEL_ORDER.index(row['model']) if row['model'] in MODEL_ORDER else 99
    s = (STRATEGY_ORDER.index(row['strategy'])
         if row['strategy'] in STRATEGY_ORDER else 99)
    return (m, s, not row['use_rr'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='without this the script only reports')
    a = ap.parse_args()

    dirs = seed_dirs()
    if not dirs:
        sys.exit(f'no prediction directories under {C.OUT_DIR}')

    total_added = 0
    for seed in sorted(dirs):
        combos = discover(dirs[seed])
        complete = {k: v for k, v in combos.items() if len(v) == N_FOLDS}
        partial = {k: v for k, v in combos.items() if 0 < len(v) < N_FOLDS}

        path = (os.path.join(C.OUT_DIR, 'grid_summary.json') if seed == 0
                else os.path.join(C.OUT_DIR, f'grid_summary_seed{seed}.json'))
        old = json.load(open(path)) if os.path.exists(path) else []
        old_keys = {(r['model'], r['strategy'], bool(r['use_rr']))
                    for r in old}

        rows = sorted((summarise(k, complete[k], seed) for k in complete),
                      key=order)
        new_keys = {(r['model'], r['strategy'], bool(r['use_rr']))
                    for r in rows}

        added = new_keys - old_keys
        lost = old_keys - new_keys

        print(f'\nseed {seed}  —  {os.path.basename(path)}')
        print(f'  on disk : {len(combos)} configurations '
              f'({len(complete)} complete, {len(partial)} partial)')
        print(f'  summary : {len(old)} rows -> {len(rows)} rows')
        for k in sorted(added):
            print(f'    + {k[0]:4} {k[1]:13} {"rr" if k[2] else "norr"}'
                  '   (was missing from the summary)')
        for k in sorted(lost):
            print(f'    ! {k[0]:4} {k[1]:13} {"rr" if k[2] else "norr"}'
                  '   in the summary but NOT on disk — not rebuilt')
        for k in sorted(partial):
            print(f'    ~ {k[0]:4} {k[1]:13} {"rr" if k[2] else "norr"}'
                  f'   only {len(partial[k])}/{N_FOLDS} folds — skipped')
        total_added += len(added)

        if a.write:
            with open(path, 'w') as fh:
                json.dump(rows, fh, indent=2)
            print(f'  written')

    print('\n' + '=' * 72)
    if not a.write:
        print(f'DRY RUN — nothing written. {total_added} missing row(s) '
              'would be restored.')
        print('Re-run with --write to apply, then run seed_report.py again.')
    else:
        print('Summaries rebuilt. Run seed_report.py again — any number it')
        print('reports may change, because it now sees every seed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())