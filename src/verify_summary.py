"""
Recompute every row of every grid summary from the prediction files.

    python src/verify_summary.py

grid_summary.json is written by run_grid.py at the end of a run. The
prediction files under artifacts/preds*/ are written per fold, as each
fold finishes, and are reused on a resume. Those two can drift apart:
if a summary row was written before a fix and the folds were recomputed
afterwards (or vice versa), nothing errors and nothing looks wrong --
the JSON simply describes an older state of the predictions than the
.npz files do.

Every number quoted in the paper comes from these summaries, so this
script recomputes each row straight from the .npz files and prints any
row that disagrees. It writes nothing and changes nothing.

Exit status is 1 if any row mismatches, so it can be used as a check.
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

N_FOLDS = 5
TOL = 1e-6          # these are recomputations, not approximations


def pred_dir_for(seed):
    return (os.path.join(C.OUT_DIR, 'preds') if seed == 0
            else os.path.join(C.OUT_DIR, f'preds_seed{seed}'))


def recompute(directory, model, strategy, use_rr):
    """Reproduce run_grid.py's two reported numbers from the fold files.

    macro_f1_mean is the MEAN OVER FOLDS; per_class_f1 is computed on the
    pooled beats. They are different reductions and run_grid.py reports
    both, so both are reproduced the same way here.
    """
    rr = 'rr' if use_rr else 'norr'
    per_fold, Y, P = [], [], []
    for f in range(N_FOLDS):
        p = os.path.join(directory, f'{model}_{strategy}_{rr}_f{f}.npz')
        if not os.path.exists(p):
            return None
        d = np.load(p)
        y, pred = d['y_true'], d['proba'].argmax(1)
        per_fold.append(f1_score(y, pred, average='macro',
                                 labels=C.EVAL_LABELS, zero_division=0))
        Y.append(y)
        P.append(pred)
    pooled = f1_score(np.concatenate(Y), np.concatenate(P), average=None,
                      labels=list(range(len(C.CLASS_NAMES))), zero_division=0)
    return (float(np.mean(per_fold)),
            {c: float(pooled[i]) for i, c in enumerate(C.CLASS_NAMES)})


def main():
    paths = []
    base = os.path.join(C.OUT_DIR, 'grid_summary.json')
    if os.path.exists(base):
        paths.append((0, base))
    for p in sorted(glob.glob(os.path.join(C.OUT_DIR,
                                           'grid_summary_seed*.json'))):
        s = os.path.basename(p).replace('grid_summary_seed', '').replace(
            '.json', '')
        paths.append((int(s), p))

    if not paths:
        sys.exit(f'no grid summaries found in {C.OUT_DIR}')

    bad = skipped = checked = 0
    for seed, path in paths:
        rows = json.load(open(path))
        d = pred_dir_for(seed)
        print(f'\n{os.path.basename(path)}  ({len(rows)} rows)  vs  '
              f'{os.path.basename(d)}/')
        print('-' * 72)
        for r in rows:
            got = recompute(d, r['model'], r['strategy'], r['use_rr'])
            label = (f"{r['model']:4} {r['strategy']:13} "
                     f"{'rr' if r['use_rr'] else 'norr':4}")
            if got is None:
                print(f'  {label}  prediction files absent — skipped')
                skipped += 1
                continue
            checked += 1
            mf1, pcf1 = got
            problems = []
            if abs(mf1 - r['macro_f1_mean']) > TOL:
                problems.append(f"macro_f1_mean {r['macro_f1_mean']:.4f} "
                                f'-> {mf1:.4f}')
            for c in C.EVAL_CLASSES:
                was = r['per_class_f1'].get(c)
                if was is not None and abs(pcf1[c] - was) > TOL:
                    problems.append(f'{c} {was:.4f} -> {pcf1[c]:.4f}')
            if problems:
                bad += 1
                print(f'  {label}  STALE: ' + ';  '.join(problems))

    print('\n' + '=' * 72)
    print(f'{checked} rows checked, {skipped} skipped, {bad} stale')
    if bad:
        print('\nA stale row means the JSON was written before the')
        print('predictions it describes. The .npz files are the truth --')
        print('delete the affected grid_summary*.json and re-run')
        print('run_grid.py, which will reuse the cached folds and rewrite')
        print('the summary in seconds without retraining anything.')
    else:
        print('\nEvery summary agrees with its prediction files.')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())