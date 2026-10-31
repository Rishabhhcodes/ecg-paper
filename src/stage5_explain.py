"""
Stage 5 — explainability and efficiency.

    python src/stage5_explain.py

Answers one question the rest of the project cannot: the RR ablation shows
that removing the timing features destroys Class S performance, but it
does not show that the trained model actually leans on them. An ablation
is evidence about the feature set. This stage is evidence about the model.

**DS2 is never opened.** Everything here runs inside DS1, on beats each
model did not train on.

What it does
------------
1. SEGMENT OCCLUSION. Each beat window is split into the clinical regions
   already fixed in config.SEGMENTS -- P, QRS, T -- plus the four RR
   features as a fourth unit. One region at a time is replaced by the
   training-fold mean and the drop in the model's probability for the
   true class is recorded. A large drop means the model was relying on
   that region.

2. A FAITHFULNESS TEST. An attribution that nobody checks is decoration.
   Regions are deleted cumulatively, most-important first, and the score
   is tracked; the same is done in a random order. If our ordering does
   not degrade the model faster than chance, the attribution is not
   telling us anything. Insertion is the mirror image: start from the
   baseline and add regions back.

3. EFFICIENCY. Parameter count, model size, and per-beat latency on CPU
   and GPU -- the embedded-feasibility claim.

Why occlusion rather than gradients
-----------------------------------
Integrated gradients and friends are more fashionable and would give a
per-sample map. They also need a baseline choice, a path, a step count
and a saturation check, each of which is a decision to defend. Occlusion
over regions that were fixed in config.py before any of this was written
needs none of that: it asks what the model does when a region is taken
away, which is the question, and it is model-agnostic enough that nobody
has to trust an implementation detail. The cost is resolution, and at
four regions we do not need more.

The baseline is the TRAINING FOLD MEAN, per feature, not zero. The
waveform is per-record z-scored so zero is close to the record mean, but
the RR features are raw seconds and ratios where zero is not a neutral
value, it is an impossible heart rate. A mean baseline is neutral for
both.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

N_FOLDS = 5
OUT = os.path.join(C.OUT_DIR, 'stage5')

# P, QRS and T come from config.SEGMENTS, fixed before this stage existed.
# 'RR' is not a slice of the window -- it is the four timing features that
# bypass the convolutional stack entirely -- so it is handled separately
# and named here so the two kinds of unit are never confused.
RR_UNIT = 'RR'


def units():
    return list(C.SEGMENTS.keys()) + [RR_UNIT]


def load_folds(y, pid):
    f = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folds.json')
    if not os.path.exists(f):
        raise SystemExit(f'{f} is missing — pull main. Do NOT run make_folds.py.')
    with open(f) as fh:
        spec = json.load(fh)
    assert spec['n_beats'] == len(y), 'folds.json does not match this ds1.npz'
    pf = {str(k): int(v) for k, v in spec['patient_fold'].items()}
    a = np.array([pf[str(p)] for p in pid])
    idx = np.arange(len(y))
    return [(idx[a != k], idx[a == k]) for k in range(spec['n_folds'])]


# --------------------------------------------------------------- forward

def predict(model, X, rr, device, batch=2048):
    """Softmax probabilities, in eval mode so BatchNorm uses running stats."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            rb = torch.tensor(rr[i:i + batch], dtype=torch.float32, device=device)
            out.append(torch.softmax(model(xb, rb), 1).cpu().numpy())
    return np.vstack(out)


def occlude(X, rr, unit, base_X, base_rr):
    """Return copies with one unit replaced by the baseline."""
    Xo, rro = X.copy(), rr.copy()
    if unit == RR_UNIT:
        rro[:, :] = base_rr
    else:
        lo, hi = C.SEGMENTS[unit]
        Xo[:, lo:hi] = base_X[lo:hi]
    return Xo, rro


# ----------------------------------------------------------- attribution

def attribution(model, X, rr, y, base_X, base_rr, device):
    """Mean drop in P(true class) when each unit is replaced, per class.

    Measured on the TRUE class rather than the predicted one: we want to
    know which region carries the evidence for what the beat actually is,
    including on beats the model gets wrong.
    """
    p0 = predict(model, X, rr, device)
    idx = np.arange(len(y))
    base_true = p0[idx, y]

    per_unit = {}
    for u in units():
        Xo, rro = occlude(X, rr, u, base_X, base_rr)
        p1 = predict(model, Xo, rro, device)
        per_unit[u] = base_true - p1[idx, y]          # positive = relied on

    out = {}
    for c in C.EVAL_CLASSES:
        m = y == C.CLASS_TO_INT[c]
        if not m.any():
            continue
        out[c] = {'n': int(m.sum()),
                  **{u: float(per_unit[u][m].mean()) for u in units()}}
    out['ALL'] = {'n': int(len(y)),
                  **{u: float(per_unit[u].mean()) for u in units()}}
    return out, per_unit


# ---------------------------------------------------------- faithfulness

def curves(model, X, rr, y, base_X, base_rr, device, order, cls):
    """Deletion and insertion curves FOR ONE CLASS.

    Deletion starts from the intact beat and removes units one at a time;
    insertion starts from an all-baseline input and adds them back. The
    quantity tracked is the mean P(cls) over the beats that really are
    cls.

    Restricting both the ordering and the measurement to one class is not
    a refinement, it is the difference between a valid test and an
    invalid one. The first version of this function pooled every class
    and ordered the units by the pooled attribution. Two things went
    wrong and they compounded:

      - 89% of beats are class N, whose attributions are near zero, so
        the pooled ordering was noise. Fifteen runs produced nine
        different orderings, and the test then compared noise against
        random.

      - With every region masked the model falls back on its prior, which
        is class N. Pooled P(true class) therefore RECOVERED at the fully
        masked end: the deletion curve was U-shaped, ending higher than
        it started in 12 of 15 runs, so its area measured almost nothing.

    Per class, neither happens: the ordering comes from a signal that is
    actually there, and masking everything drives P(cls) toward the prior
    for cls, which for a minority class is near zero.
    """
    i = C.CLASS_TO_INT[cls]
    m = y == i
    if m.sum() < 20:
        return None, None
    Xm, rrm = X[m], rr[m]

    def mean_p(Xc, rrc):
        return float(predict(model, Xc, rrc, device)[:, i].mean())

    dele, Xc, rrc = [mean_p(Xm, rrm)], Xm.copy(), rrm.copy()
    for u in order:
        Xc, rrc = occlude(Xc, rrc, u, base_X, base_rr)
        dele.append(mean_p(Xc, rrc))

    Xi, rri = Xm.copy(), rrm.copy()
    for u in units():                                  # start fully masked
        Xi, rri = occlude(Xi, rri, u, base_X, base_rr)
    ins = [mean_p(Xi, rri)]
    for u in order:
        if u == RR_UNIT:
            rri = rrm.copy()
        else:
            lo, hi = C.SEGMENTS[u]
            Xi[:, lo:hi] = Xm[:, lo:hi]
        ins.append(mean_p(Xi, rri))
    return dele, ins


def auc(vals):
    """Normalised area under a curve sampled at equal steps."""
    return float(np.trapezoid(vals, dx=1.0) / (len(vals) - 1)) if len(vals) > 1 \
        else float(vals[0])


# ------------------------------------------------------------ efficiency

def efficiency(model, X, rr, device):
    import torch
    from models import count_params

    params = int(count_params(model))
    size_mb = sum(p.numel() * p.element_size()
                  for p in model.parameters()) / 1e6

    out = {'parameters': params, 'size_mb': round(size_mb, 3)}
    xs, rs = X[:1024], rr[:1024]

    for dev in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
        m = model.to(dev).eval()
        xb = torch.tensor(xs, dtype=torch.float32, device=dev)
        rb = torch.tensor(rs, dtype=torch.float32, device=dev)
        with torch.no_grad():
            for _ in range(3):                         # warm up
                m(xb[:128], rb[:128])
            if dev == 'cuda':
                torch.cuda.synchronize()
            t = time.perf_counter()
            for i in range(0, 1024, 128):
                m(xb[i:i + 128], rb[i:i + 128])
            if dev == 'cuda':
                torch.cuda.synchronize()
            batch_s = time.perf_counter() - t

            t = time.perf_counter()
            for i in range(64):                        # one beat at a time
                m(xb[i:i + 1], rb[i:i + 1])
            if dev == 'cuda':
                torch.cuda.synchronize()
            single_s = time.perf_counter() - t

        out[f'{dev}_ms_per_beat_batched'] = round(batch_s / 1024 * 1000, 4)
        out[f'{dev}_ms_per_beat_single'] = round(single_s / 64 * 1000, 4)
    model.to(device)
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='class_weight',
                    help='class_weight or focal — Stage 2 could not separate '
                         'them, so either is defensible; both are reported '
                         'in the paper')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=60)
    a = ap.parse_args()

    import torch
    from train_cnn import train_cnn

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUT, exist_ok=True)

    p = os.path.join(C.OUT_DIR, 'ds1.npz')
    if not os.path.exists(p):
        raise SystemExit(f'missing {p} — build or upload the dataset first')
    d = np.load(p)
    X, rr, y, pid = d['X'], d['rr'], d['y'], d['pid']
    folds = load_folds(y, pid)

    print('=' * 74)
    print(f'STAGE 5 — EXPLAINABILITY AND EFFICIENCY   (cnn / {a.strategy} / rr)')
    print('=' * 74)
    print(f'DS1 {len(y):,} beats / {len(np.unique(pid))} patients   device {device}')
    print(f'units: {units()}   '
          + '  '.join(f'{k}{v}' for k, v in C.SEGMENTS.items()))
    print('DS2 is not opened by this stage.\n')

    runs, eff = [], None
    for seed in a.seeds:
        for f, (tr, va) in enumerate(folds):
            t0 = time.time()
            _, meta = train_cnn(X[tr], rr[tr], y[tr], X[va], rr[va], y[va],
                                use_rr=True, strategy=a.strategy,
                                epochs=a.epochs, seed=seed, verbose=False,
                                return_model=True)
            model = meta.pop('model')

            # Baseline from the TRAINING fold only — a baseline computed on
            # the beats being explained would leak them into the explanation.
            base_X = X[tr].mean(0)
            base_rr = rr[tr].mean(0)

            att, per_unit = attribution(model, X[va], rr[va], y[va],
                                        base_X, base_rr, device)

            # Ordering and measurement are both per class. Ordering on the
            # pooled row put class N -- 89% of beats, attributions near
            # zero -- in charge of the ordering, which made it noise.
            rng = np.random.default_rng(seed * 100 + f)
            faith = {}
            for c in C.EVAL_CLASSES:
                if c not in att:
                    continue
                order = sorted(units(), key=lambda u: -att[c][u])
                rand = list(rng.permutation(units()))
                de, ii = curves(model, X[va], rr[va], y[va],
                                base_X, base_rr, device, order, c)
                if de is None:                       # too few beats of c
                    continue
                dr_, ir_ = curves(model, X[va], rr[va], y[va],
                                  base_X, base_rr, device, rand, c)
                faith[c] = {'order': order, 'random_order': rand,
                            'deletion': de, 'insertion': ii,
                            'deletion_random': dr_, 'insertion_random': ir_}

            if eff is None:
                eff = efficiency(model, X[va], rr[va], device)

            runs.append({'seed': seed, 'fold': f, 'attribution': att,
                         'faithfulness': faith,
                         'best_epoch': meta.get('best_epoch')})
            tops = '  '.join(f'{c}:{max(units(), key=lambda u: att[c][u])}'
                             for c in C.EVAL_CLASSES if c in att)
            print(f'  seed {seed} fold {f}  top unit  {tops:34}'
                  f'   [{time.time() - t0:.0f}s]', flush=True)
            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    # ------------------------------------------------------------ report
    print('\n' + '=' * 74)
    print('WHAT THE MODEL LEANS ON — mean drop in P(true class) when a')
    print('region is replaced by the training-fold mean')
    print('=' * 74)
    U = units()
    print(f"{'class':6} {'n':>7}  " + '  '.join(f'{u:>9}' for u in U))
    print('-' * 74)
    for c in C.EVAL_CLASSES + ['ALL']:
        vals = [[r['attribution'][c][u] for r in runs if c in r['attribution']]
                for u in U]
        if not vals[0]:
            continue
        n = int(np.mean([r['attribution'][c]['n'] for r in runs
                         if c in r['attribution']]))
        cells = []
        best = max(range(len(U)), key=lambda i: np.mean(vals[i]))
        for i, u in enumerate(U):
            m = np.mean(vals[i])
            cells.append(f'{m:>9.4f}' + ('*' if i == best else ' '))
        print(f'{c:6} {n:>7,}  ' + ' '.join(cells))
    print('\n* = largest drop for that class.')

    print('\n' + '=' * 74)
    print('IS THE ATTRIBUTION FAITHFUL?')
    print('=' * 74)
    print('Deleting regions in attributed order should degrade the model')
    print('faster than deleting them at random. Lower deletion AUC is better;')
    print('higher insertion AUC is better.\n')
    from collections import Counter
    print(f"{'class':6} {'runs':>5}  {'deletion':>21}  {'insertion':>21}")
    print(f"{'':6} {'':>5}  {'attrib  random  wins':>21}"
          f"  {'attrib  random  wins':>21}")
    print('-' * 74)
    for c in C.EVAL_CLASSES:
        rs = [r['faithfulness'][c] for r in runs if c in r['faithfulness']]
        if not rs:
            continue
        da = [auc(x['deletion']) for x in rs]
        dr = [auc(x['deletion_random']) for x in rs]
        ia = [auc(x['insertion']) for x in rs]
        ir = [auc(x['insertion_random']) for x in rs]
        wd = sum(x < z for x, z in zip(da, dr))
        wi = sum(x > z for x, z in zip(ia, ir))
        print(f'{c:6} {len(rs):>5}  {np.mean(da):6.4f}  {np.mean(dr):6.4f}'
              f'  {wd:>2}/{len(rs):<2}  {np.mean(ia):6.4f}  {np.mean(ir):6.4f}'
              f'  {wi:>2}/{len(rs):<2}')
    print('\n  Lower deletion and higher insertion than random means the')
    print('  attributed ordering is informative for that class.')
    for c in C.EVAL_CLASSES:
        rs = [r['faithfulness'][c] for r in runs if c in r['faithfulness']]
        if not rs:
            continue
        oc = Counter(' > '.join(x['order']) for x in rs)
        o, k = oc.most_common(1)[0]
        print(f'    {c}: most common ordering {o}  ({k}/{len(rs)} runs)')

    print('\n' + '=' * 74)
    print('EFFICIENCY')
    print('=' * 74)
    for k, v in eff.items():
        print(f'  {k:28} {v}')

    res = {'configuration': {'model': 'cnn', 'strategy': a.strategy,
                             'use_rr': True},
           'segments': {k: list(v) for k, v in C.SEGMENTS.items()},
           'seeds': a.seeds, 'units': U, 'efficiency': eff, 'runs': runs}
    with open(os.path.join(OUT, 'stage5_results.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    make_figures(res, OUT)
    print(f'\nWrote stage5_results.json, attribution.png, faithfulness.png '
          f'-> {OUT}')
    print('DS2 was not opened.')
    return 0


# -------------------------------------------------------------- figures

def make_figures(res, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    U, runs = res['units'], res['runs']
    classes = [c for c in C.EVAL_CLASSES
               if c in runs[0]['attribution']] + ['ALL']

    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout='constrained')
    w = 0.8 / len(U)
    colours = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']
    for i, u in enumerate(U):
        means = [np.mean([r['attribution'][c][u] for r in runs
                          if c in r['attribution']]) for c in classes]
        errs = [(lambda v: (max(v) - min(v)) / 2)(
                [r['attribution'][c][u] for r in runs
                 if c in r['attribution']]) for c in classes]
        ax.bar([x + i * w for x in range(len(classes))], means, w,
               yerr=errs, capsize=2, label=u, color=colours[i % len(colours)])
    ax.set_xticks([x + 0.4 - w / 2 for x in range(len(classes))])
    ax.set_xticklabels(classes)
    ax.set_xlabel('true class')
    ax.set_ylabel('drop in P(true class) when the region is removed')
    ax.set_title('What the CNN leans on, by region and class\n'
                 f'{len(runs)} runs; bars are min-max across runs', fontsize=10)
    ax.axhline(0, color='k', lw=0.8)
    ax.legend(fontsize=9, title='region')
    ax.grid(axis='y', alpha=0.25)
    fig.savefig(os.path.join(outdir, 'attribution.png'), dpi=150)
    plt.close(fig)

    present = [c for c in C.EVAL_CLASSES
               if any(c in r['faithfulness'] for r in runs)]
    xs = list(range(len(U) + 1))
    fig, axes = plt.subplots(2, len(present),
                             figsize=(3.1 * len(present), 6.2),
                             layout='constrained', squeeze=False)
    for j, c in enumerate(present):
        rs = [r['faithfulness'][c] for r in runs if c in r['faithfulness']]
        for i, (key, rkey, title) in enumerate((
                ('deletion', 'deletion_random', 'deletion'),
                ('insertion', 'insertion_random', 'insertion'))):
            ax = axes[i][j]
            A = np.array([x[key] for x in rs])
            B = np.array([x[rkey] for x in rs])
            ax.plot(xs, A.mean(0), 'o-', color='#4C78A8', ms=4,
                    label='attributed')
            ax.fill_between(xs, A.min(0), A.max(0), color='#4C78A8',
                            alpha=0.15, lw=0)
            ax.plot(xs, B.mean(0), 's--', color='#999999', ms=4,
                    label='random')
            ax.fill_between(xs, B.min(0), B.max(0), color='#999999',
                            alpha=0.15, lw=0)
            ax.set_title(f'class {c} — {title}', fontsize=9)
            ax.set_xlabel('regions ' + ('removed' if i == 0 else 'restored'),
                          fontsize=8)
            if j == 0:
                ax.set_ylabel(f'mean P({c})', fontsize=8)
            ax.set_xticks(xs)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle('Faithfulness, per class — does the attributed ordering '
                 'matter?\nordering and score both restricted to the class',
                 fontsize=10)
    fig.savefig(os.path.join(outdir, 'faithfulness.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())