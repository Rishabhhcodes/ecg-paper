"""
Generate the frozen fold assignment for DS1.

Run this ONCE, commit the resulting src/folds.json, and never run it again
unless the protocol itself changes. Regenerating it invalidates every
prediction file in the project.

Why this file exists
--------------------
run_grid.py used to build the folds by calling StratifiedGroupKFold on every
run. That is deterministic for a given scikit-learn version, but NOT across
versions. Two halves of the Stage 2 grid were produced on

    laptop : scikit-learn 1.9.1
    Colab  : scikit-learn 1.6.1

and the two machines partitioned the same 22 patients differently — fold 0
held 10,467 validation beats on one and 10,731 on the other. Nothing failed
loudly; the LR/XGBoost rows and the CNN rows simply described different
partitions of DS1, which makes comparing them meaningless.

The fold assignment is a property of the experiment, not an output of a
library. So it is computed once and distributed as data.

The file records the assignment PER PATIENT, not per beat. That makes it 22
lines instead of 50,977, readable in a diff, and makes the patient-disjoint
property self-evident to anyone reviewing it.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

N_FOLDS = 5
SEED = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folds.json')


def main():
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    if os.path.exists(OUT):
        print(f'{OUT} already exists.')
        print('Regenerating it would invalidate every prediction file in')
        print('artifacts/preds/. Delete it by hand first if that is really')
        print('what you want.')
        return 1

    d = np.load(os.path.join(C.OUT_DIR, 'ds1.npz'))
    y, pid = d['y'], d['pid']

    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
    folds = list(gkf.split(np.zeros(len(y)), y, groups=pid))

    # Every patient must sit entirely inside one fold. This is the property
    # the whole inter-patient protocol rests on, so it is asserted rather
    # than assumed.
    patient_fold = {}
    for f, (tr, va) in enumerate(folds):
        overlap = set(pid[tr].tolist()) & set(pid[va].tolist())
        assert not overlap, f'fold {f}: patient overlap {sorted(overlap)}'
        for p in np.unique(pid[va]):
            assert str(p) not in patient_fold, f'patient {p} in two folds'
            patient_fold[str(p)] = f

    assert len(patient_fold) == len(np.unique(pid)), 'some patient has no fold'

    # Reconstruct the folds from the per-patient map and confirm the result
    # is identical to what the splitter produced. If this fails, the file
    # would silently describe a different experiment.
    assign = np.array([patient_fold[str(p)] for p in pid])
    idx = np.arange(len(y))
    for f, (tr, va) in enumerate(folds):
        assert np.array_equal(idx[assign == f], va), f'fold {f} val mismatch'
        assert np.array_equal(idx[assign != f], tr), f'fold {f} train mismatch'

    spec = {
        'dataset': 'ds1',
        'n_folds': N_FOLDS,
        'seed': SEED,
        'n_beats': int(len(y)),
        'n_patients': int(len(np.unique(pid))),
        'splitter': 'StratifiedGroupKFold(shuffle=True)',
        'generated_with': {
            'python': sys.version.split()[0],
            'sklearn': sklearn.__version__,
            'numpy': np.__version__,
        },
        'note': ('Frozen on purpose. StratifiedGroupKFold is not stable '
                 'across scikit-learn versions; this file is the protocol. '
                 'Do not regenerate.'),
        'patient_fold': {k: patient_fold[k]
                         for k in sorted(patient_fold, key=int)},
    }

    with open(OUT, 'w') as fh:
        json.dump(spec, fh, indent=2)

    print(f'wrote {OUT}')
    print()
    for f, (tr, va) in enumerate(folds):
        pats = sorted(int(p) for p in np.unique(pid[va]))
        counts = np.bincount(y[va], minlength=5)
        print(f'  fold {f}: {len(va):>6,} val beats  '
              f'{dict(zip(C.CLASS_NAMES, counts.tolist()))}')
        print(f'           patients {pats}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
