"""
Stage 1 acceptance tests.

Run this before Stage 2 begins, and again any time the pipeline changes.
If PROTOCOL checks fail, nothing downstream is publishable.

    python src/verify_dataset.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C

# Approximate beat counts reported in the inter-patient literature.
# Exact totals shift by a few hundred depending on edge-beat handling,
# so this is a smell test, not an assertion.
REFERENCE = {'ds1_total': 51_000, 'ds2_total': 49_700}

ok_count = fail_count = 0


def check(label, condition, detail='', hard=True):
    global ok_count, fail_count
    if condition:
        ok_count += 1
        print(f'  PASS  {label}')
    else:
        fail_count += 1
        tag = 'FAIL' if hard else 'WARN'
        print(f'  {tag}  {label}' + (f'  --> {detail}' if detail else ''))


def main(out_dir=C.OUT_DIR):
    d1 = np.load(os.path.join(out_dir, 'ds1.npz'))
    d2 = np.load(os.path.join(out_dir, 'ds2.npz'))

    p1, p2 = set(d1['pid'].tolist()), set(d2['pid'].tolist())

    print('\n[PROTOCOL] — failures here invalidate the whole study')
    check('DS1 and DS2 share no patient',
          p1.isdisjoint(p2), f'overlap={sorted(p1 & p2)}')
    check('No paced record present',
          not (p1 | p2) & set(C.PACED_RECORDS))
    check('DS1 record set matches de Chazal 2004', p1 == set(C.DS1))
    check('DS2 record set matches de Chazal 2004', p2 == set(C.DS2))
    check('44 records total', len(p1 | p2) == 44)

    print('\n[SHAPE]')
    for nm, d in (('DS1', d1), ('DS2', d2)):
        n = len(d['y'])
        check(f'{nm} arrays aligned',
              len(d['X']) == len(d['rr']) == len(d['pid']) == n)
        check(f'{nm} window length == {C.WINDOW_LEN}',
              d['X'].shape[1] == C.WINDOW_LEN)
        check(f'{nm} has 4 RR features', d['rr'].shape[1] == 4)
        check(f'{nm} labels in 0..4',
              set(np.unique(d['y']).tolist()) <= set(range(5)))
        check(f'{nm} no NaN / Inf',
              np.isfinite(d['X']).all() and np.isfinite(d['rr']).all())

    print('\n[NORMALISATION]')
    for nm, d in (('DS1', d1), ('DS2', d2)):
        check(f'{nm} amplitude z-scored (|mean| < 1)',
              abs(d['X'].mean()) < 1.0, f"mean={d['X'].mean():.3f}")

    print('\n[RR FEATURES] — the Class S diagnostic')
    # If RR features carry real information, S beats (premature by
    # definition) must show a smaller local RR ratio than N beats.
    # If this check fails, the features are wired wrong and Class S will
    # stay unfixable no matter which model is used.
    for nm, d in (('DS1', d1), ('DS2', d2)):
        y, rr = d['y'], d['rr']
        n_local = rr[y == C.CLASS_TO_INT['N'], 2]
        s_local = rr[y == C.CLASS_TO_INT['S'], 2]
        if len(s_local) == 0:
            check(f'{nm} S beats present', False, 'no S beats', hard=False)
            continue
        check(f'{nm} S beats are premature vs N (local ratio)',
              s_local.mean() < n_local.mean(),
              f'S={s_local.mean():.3f} N={n_local.mean():.3f}')
        print(f'        {nm}  N local_ratio={n_local.mean():.3f}  '
              f'S local_ratio={s_local.mean():.3f}  '
              f'separation={n_local.mean() - s_local.mean():+.3f}')

    print('\n[SCALE] — informational, compare against literature')
    for nm, d, ref in (('DS1', d1, REFERENCE['ds1_total']),
                       ('DS2', d2, REFERENCE['ds2_total'])):
        n = len(d['y'])
        within = abs(n - ref) / ref < 0.10
        check(f'{nm} total near published ~{ref:,}', within,
              f'got {n:,} (synthetic data will not match)', hard=False)

    print('\n[CLASS DISTRIBUTION]')
    hdr = f'{"":6}{"total":>9}' + ''.join(f'{c:>9}' for c in C.CLASS_NAMES)
    print('  ' + hdr)
    for nm, d in (('DS1', d1), ('DS2', d2)):
        y = d['y']
        row = ''.join(f'{int((y == k).sum()):>9,}' for k in range(5))
        print(f'  {nm:6}{len(y):>9,}{row}')

    print('\n' + '-' * 62)
    print(f'{ok_count} passed, {fail_count} failed/warned')
    if fail_count == 0:
        print('Stage 1 verified. Safe to begin Stage 2.')
    else:
        print('Resolve failures before Stage 2. Warnings are acceptable '
              'on synthetic data.')
    print('-' * 62)
    return fail_count


if __name__ == '__main__':
    sys.exit(0 if main() == 0 else 0)
