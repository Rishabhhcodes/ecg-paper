"""
Aggregate the grid across seeds and say which comparisons are stable.

Run after the multi-seed grid:

    python src/seed_report.py

It reads artifacts/grid_summary.json (seed 0) and every
artifacts/grid_summary_seed*.json, then reports, for each configuration,
the mean and spread across seeds.

Why this exists
---------------
One configuration was run twice on identical code, identical data,
identical settings and the same fold assignment. macro-F1 came out 0.4920
then 0.4893; Class S F1 came out 0.216 then 0.397. The cause is GPU
non-determinism interacting with early stopping: a slightly different
trajectory selects a different epoch, and the rare-class score is very
sensitive to which epoch is kept.

That single observation told us the order of magnitude but not much else.
This script replaces it with a measured spread, and then does the thing
the spread is actually for: it checks every comparison we want to make in
the paper and says whether the gap survives the noise.

A difference smaller than the run-to-run spread is not a finding. The
point of this report is to stop us writing one.
"""
import glob
import json
import os
import sys
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ART = C.OUT_DIR


def load():
    """Return {(model, strategy, use_rr): {seed: row}}."""
    paths = []
    base = os.path.join(ART, 'grid_summary.json')
    if os.path.exists(base):
        paths.append(base)
    paths += sorted(glob.glob(os.path.join(ART, 'grid_summary_seed*.json')))

    if not paths:
        raise SystemExit(f'no grid summaries found in {ART}')

    runs = {}
    for pth in paths:
        with open(pth) as fh:
            rows = json.load(fh)
        for r in rows:
            # Older summaries predate the seed field; those are seed 0.
            s = int(r.get('seed', 0))
            runs.setdefault((r['model'], r['strategy'], bool(r['use_rr'])),
                            {})[s] = r

    seeds = sorted({s for v in runs.values() for s in v})
    print(f'read {len(paths)} summaries, seeds {seeds}')

    # That seed list is the UNION over every configuration, which is not
    # the same thing as every configuration having every seed -- and once
    # it was not. run_grid.py rebuilds its summary from scratch on each
    # run, so a later `--models lr xgb` replaced a 26-row
    # grid_summary.json with a 16-row one and took seed 0's ten CNN rows
    # with it. The predictions were still on disk. This report averaged
    # the CNN over four seeds while printing "seeds [0, 1, 2, 3, 4]",
    # which it had read from the LR and XGBoost rows.
    #
    # Every mean below is only as good as the seeds behind it, so a
    # configuration missing one is named before anything is reported.
    #
    # Compared against the seeds that model actually has, not the global
    # union. Only the CNN was repeated -- logistic regression is
    # deterministic and XGBoost moves by under 0.001 -- so measuring LR
    # and XGBoost against the union would report all sixteen of their
    # rows as missing four seeds on every single run. A warning that
    # fires when nothing is wrong is a warning nobody reads.
    model_seeds = {}
    for (m, _, _), v in runs.items():
        model_seeds.setdefault(m, set()).update(v)
    short = {k: sorted(model_seeds[k[0]] - set(v)) for k, v in runs.items()
             if len(v) < len(model_seeds[k[0]])}
    if short:
        print(f'\n*** {len(short)} configuration(s) are missing a seed. '
              'Their means and spreads\n*** below are computed over fewer '
              'runs than that header implies.')
        for k in sorted(short):
            print(f'      {k[0]:4} {k[1]:13} {"rr" if k[2] else "norr":4} '
                  f'   missing seed(s) {short[k]}')
        print('*** Fix: python src/rebuild_summary.py --write, then re-run '
              'this.')
    print()
    return runs, seeds


def val(row, what):
    return (row['macro_f1_mean'] if what == 'macro-F1'
            else row['per_class_f1'][what])


def agg(runs, key, what):
    """(mean, spread, n) across seeds for one configuration."""
    rows = runs.get(key)
    if not rows:
        return None
    vals = [val(r, what) for r in rows.values()]
    return (mean(vals), (max(vals) - min(vals)) if len(vals) > 1 else 0.0,
            len(vals))


def main():
    runs, seeds = load()
    if len(seeds) < 2:
        print('Only one seed present. Run the grid again with --seed 1 '
              '(and 2, 3, 4) before this report means anything.')
        return 1

    # ---------------------------------------------------------------- 1
    print('=' * 74)
    print('PER-CONFIGURATION, ACROSS SEEDS')
    print('=' * 74)
    print(f"{'model':5} {'strategy':13} {'rr':>2}   "
          f"{'macro-F1':>16}   {'S':>16}")
    print('-' * 74)

    for key in sorted(runs, key=lambda k: (k[0], k[1], not k[2])):
        m, s, rr = key
        a = agg(runs, key, 'macro-F1')
        b = agg(runs, key, 'S')
        if a is None:
            continue
        print(f'{m:5} {s:13} {int(rr):>2}   '
              f'{a[0]:.4f} (±{a[1] / 2:.4f})   '
              f'{b[0]:.4f} (±{b[1] / 2:.4f})')

    # ---------------------------------------------------------------- 2
    print()
    print('=' * 74)
    print('RUN-TO-RUN SPREAD, BY MODEL')
    print('=' * 74)
    print('Half the range across seeds, averaged over that model\'s rows.')
    print('This is the floor below which a difference is not a finding.\n')

    for m in ('lr', 'xgb', 'cnn'):
        keys = [k for k in runs if k[0] == m and len(runs[k]) > 1]
        if not keys:
            continue
        for what in ('macro-F1', 'S'):
            spreads = [agg(runs, k, what)[1] / 2 for k in keys]
            print(f'  {m:4} {what:9} ±{mean(spreads):.4f}   '
                  f'(worst row ±{max(spreads):.4f})')
    print()

    # ---------------------------------------------------------------- 3
    print('=' * 74)
    print('RR ABLATION, AVERAGED OVER SEEDS')
    print('=' * 74)

    ok = tot = 0
    for m in ('lr', 'xgb', 'cnn'):
        for s in ('none', 'class_weight', 'focal', 'adasyn', 'smote'):
            hi, lo = agg(runs, (m, s, True), 'macro-F1'), agg(runs, (m, s, False), 'macro-F1')
            if not hi or not lo:
                continue
            for what in ('macro-F1', 'S'):
                a = agg(runs, (m, s, True), what)
                b = agg(runs, (m, s, False), what)
                gap = a[0] - b[0]
                noise = (a[1] + b[1]) / 2          # combined half-ranges
                tot += 1
                good = gap > 0
                ok += good
                verdict = ('pass' if gap > max(noise, 1e-9)
                           else ('pass (inside noise)' if good else '*** FAIL ***'))
                print(f'  {m:4} {s:13} {what:9} '
                      f'{a[0]:.4f} vs {b[0]:.4f}  gap {gap:+.4f}  '
                      f'noise ±{noise:.4f}   {verdict}')
    print(f'\n  {ok}/{tot} comparisons in the right direction')

    # ---------------------------------------------------------------- 4
    print()
    print('=' * 74)
    print('CLOSE COMPARISONS — can we rank these or not?')
    print('=' * 74)
    print('These are the claims that needed multiple seeds to settle.\n')

    pairs = [
        ('cnn', 'class_weight', 'cnn', 'focal'),
        ('cnn', 'smote', 'cnn', 'adasyn'),
        ('cnn', 'class_weight', 'cnn', 'smote'),
        ('xgb', 'smote', 'xgb', 'class_weight'),
        ('xgb', 'class_weight', 'xgb', 'adasyn'),
    ]
    # Checked on Class S as well as macro-F1. This section used to report
    # macro-F1 only, and the claim that mattered most -- that focal loss
    # beats class weighting on Class S, the class this whole project is
    # about -- was a CLASS S claim. It therefore never passed through this
    # rule at all, and went into the README on a hand calculation over a
    # summary that was silently missing a seed. Class S is also the noisier
    # of the two by a factor of four, so it is the one that most needs it.
    for m1, s1, m2, s2 in pairs:
        print(f'  {m1} {s1} vs {s2}')
        for what in ('macro-F1', 'S'):
            a = agg(runs, (m1, s1, True), what)
            b = agg(runs, (m2, s2, True), what)
            if not a or not b:
                continue
            gap = a[0] - b[0]
            noise = (a[1] + b[1]) / 2
            n = min(a[2], b[2])
            if n < 2:
                verdict = (f'only {n} seed — noise unknown, NOT rankable '
                           '(this is not a tie)')
            elif abs(gap) > noise:
                verdict = f'RANKABLE — {s1 if gap > 0 else s2} is ahead'
            else:
                verdict = 'NOT rankable — report as indistinguishable'
            print(f'      {what:9} {a[0]:.4f} vs {b[0]:.4f}   '
                  f'gap {gap:+.4f}   noise ±{noise:.4f}')
            print(f'      {"":9} {verdict}')
        print()

    print('Rule used: a gap counts only if it exceeds the combined')
    print('half-range of the two configurations across seeds. A pair with')
    print('one seed has no measured noise and cannot be ranked either way;')
    print('that is different from a measured tie.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())