"""
Diagnostic — per-fold class distribution and per-fold scores.

Explains why DS1 cross-validation scores carry such large error bars:
S and F beats are concentrated in a handful of MIT-BIH records, so a
5-fold split over 22 patients puts very few of them in some folds.

    python fold_report.py
"""
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import config as C

ART = C.OUT_DIR
d = np.load(os.path.join(ART, 'ds1.npz'))
y, pid = d['y'], d['pid']

print('=' * 72)
print('WHERE THE MINORITY BEATS LIVE  (DS1, per record)')
print('=' * 72)
print(f'{"record":>8}{"total":>9}' + ''.join(f'{c:>8}' for c in C.CLASS_NAMES))
for p in sorted(np.unique(pid)):
    m = pid == p
    row = ''.join(f'{int(((y == k) & m).sum()):>8,}' for k in range(5))
    print(f'{p:>8}{int(m.sum()):>9,}{row}')

print()
for cls in ('S', 'F'):
    k = C.CLASS_TO_INT[cls]
    per = {int(p): int(((y == k) & (pid == p)).sum()) for p in np.unique(pid)}
    tot = sum(per.values())
    top = sorted(per.items(), key=lambda kv: -kv[1])[:3]
    share = 100 * sum(v for _, v in top) / max(tot, 1)
    print(f'Class {cls}: {tot:,} beats — top 3 records hold {share:.0f}% '
          f'({", ".join(f"{r}:{v}" for r, v in top)})')

print()
print('=' * 72)
print('FOLD COMPOSITION  (GroupKFold, 5 splits, grouped by patient)')
print('=' * 72)
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(np.zeros(len(y)), y, groups=pid))
print(f'{"fold":>5}{"train":>9}{"val":>8}' +
      ''.join(f'{"tr-" + c:>8}{"va-" + c:>8}' for c in ('S', 'F')))
for f, (tr, va) in enumerate(folds):
    cells = ''
    for c in ('S', 'F'):
        k = C.CLASS_TO_INT[c]
        cells += f'{int((y[tr] == k).sum()):>8,}{int((y[va] == k).sum()):>8,}'
    print(f'{f:>5}{len(tr):>9,}{len(va):>8,}{cells}')

print()
print('=' * 72)
print('PER-FOLD SCORES  (from saved predictions)')
print('=' * 72)
files = sorted(glob.glob(os.path.join(ART, 'preds', '*.npz')))
if not files:
    print('No prediction files found — run the grid first.')
    sys.exit(0)

configs = {}
for fp in files:
    base = os.path.basename(fp)[:-4]
    cfg, fold = base.rsplit('_f', 1)
    configs.setdefault(cfg, {})[int(fold)] = fp

for cfg in sorted(configs):
    if not cfg.endswith('_rr'):          # only the with-RR rows, keeps it short
        continue
    per_fold, s_fold = [], []
    for f in sorted(configs[cfg]):
        z = np.load(configs[cfg][f])
        yt, yp = z['y_true'], z['proba'].argmax(1)
        per_fold.append(f1_score(yt, yp, average='macro',
                                 labels=C.EVAL_LABELS, zero_division=0))
        s_fold.append(f1_score(yt, yp, average=None,
                               labels=list(range(5)),
                               zero_division=0)[C.CLASS_TO_INT['S']])
    print(f'{cfg:26} macroF1 per fold: ' +
          ' '.join(f'{v:.3f}' for v in per_fold) +
          f'   (spread {max(per_fold) - min(per_fold):.3f})')
    print(f'{"":26} S-F1    per fold: ' +
          ' '.join(f'{v:.3f}' for v in s_fold) +
          f'   (spread {max(s_fold) - min(s_fold):.3f})')