"""
Compare GroupKFold against StratifiedGroupKFold on DS1.

No training — this only inspects how the two schemes distribute the
minority classes across folds. Cheap to run, decisive about whether a
re-run of the grid is worth an hour.

Both schemes keep every patient inside a single fold, so neither leaks.
The difference is only whether class proportions are balanced while
doing it.

    python cv_compare.py
"""
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import config as C

d = np.load(os.path.join(C.OUT_DIR, 'ds1.npz'))
y, pid = d['y'], d['pid']
N_FOLDS = 5


def report(name, folds):
    print(f'\n{name}')
    print('-' * 68)
    hdr = f'{"fold":>5}{"train":>9}{"val":>8}'
    for c in ('S', 'V', 'F'):
        hdr += f'{"tr-" + c:>8}{"va-" + c:>8}'
    print(hdr)

    val_counts = {c: [] for c in ('S', 'V', 'F')}
    for f, (tr, va) in enumerate(folds):
        row = f'{f:>5}{len(tr):>9,}{len(va):>8,}'
        for c in ('S', 'V', 'F'):
            k = C.CLASS_TO_INT[c]
            ntr, nva = int((y[tr] == k).sum()), int((y[va] == k).sum())
            row += f'{ntr:>8,}{nva:>8,}'
            val_counts[c].append(nva)
        print(row)

    print()
    for c in ('S', 'V', 'F'):
        v = np.array(val_counts[c])
        empty = int((v == 0).sum())
        # Coefficient of variation: 0 means every fold holds an equal
        # share, higher means the class piles into a few folds.
        cv = v.std() / max(v.mean(), 1e-9)
        flag = '   <-- unusable' if empty or cv > 1.0 else ''
        print(f'  {c}: val counts {list(v)}  spread(CV)={cv:.2f}  '
              f'empty folds={empty}{flag}')
    return val_counts


g = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups=pid))
s = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
         .split(np.zeros(len(y)), y, groups=pid))

a = report('GroupKFold  (current)', g)
b = report('StratifiedGroupKFold  (proposed)', s)

# Leak check — neither scheme may put a patient in two folds.
print('\n' + '=' * 68)
for name, folds in (('GroupKFold', g), ('StratifiedGroupKFold', s)):
    bad = [f for f, (tr, va) in enumerate(folds)
           if set(pid[tr]) & set(pid[va])]
    print(f'{name:24} patient overlap in folds: '
          f'{bad if bad else "none — leak-free"}')

print('\nVERDICT')
print('-' * 68)
for c in ('S', 'V', 'F'):
    va, vb = np.array(a[c]), np.array(b[c])
    cva = va.std() / max(va.mean(), 1e-9)
    cvb = vb.std() / max(vb.mean(), 1e-9)
    verdict = ('better' if cvb < cva * 0.8 else
               'no real change' if cvb < cva * 1.2 else 'worse')
    print(f'  {c}: spread {cva:.2f} -> {cvb:.2f}   {verdict}')
print("\nA class whose beats come from one or two records cannot be")
print("balanced by any scheme that keeps patients intact. For such a")
print("class the cross-validated score is not interpretable, and that")
print("is a property of the database, not of the pipeline.")