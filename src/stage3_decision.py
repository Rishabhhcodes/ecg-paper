"""
Stage 3 — the decision layer. Calibration, then a reject curve.

    python src/stage3_decision.py

This script NEVER retrains and NEVER opens DS2. It reads the per-fold
validation predictions that run_grid.py already wrote to artifacts/preds/
and turns them into three figures and one JSON of numbers. It needs no
GPU, no ds1.npz, and no PyTorch. It runs in well under a minute on a
laptop CPU.

What it does, and why each step is there
----------------------------------------

1. TEMPERATURE SCALING.  Guo et al., "On Calibration of Modern Neural
   Networks" (ICML 2017), show that modern networks are badly calibrated
   -- the softmax number is not a probability you can threshold. They fit
   a single scalar T and divide the logits by it. Two properties of
   theirs matter to us:

     - "T is optimized with respect to NLL on the validation set."
     - "Because the parameter T does not change the maximum of the
        softmax function, the class prediction" is unchanged, and so
        "temperature scaling does not affect the model's accuracy."

   That second property is the important one for this project: every
   macro-F1 in Stage 2 is computed from an argmax, so calibration cannot
   move a single Stage 2 number. Nobody can claim Stage 3 was used to
   improve Stage 2.

   We stored probabilities, not logits. That is not a problem and not an
   approximation. Since p = softmax(z), log(p) = z - logsumexp(z), i.e.
   log(p) recovers z up to one additive constant shared by all classes,
   and softmax is invariant to exactly that constant:

       softmax(log(p)/T)_i = exp((z_i - K)/T) / sum_j exp((z_j - K)/T)
                           = exp(z_i/T) / sum_j exp(z_j/T)
                           = softmax(z/T)_i

   So working from the stored `proba` is identical to working from the
   logits. This holds for all three model families: multinomial logistic
   regression and XGBoost's multi:softprob are softmax outputs too.

2. T IS CROSS-FITTED ACROSS FOLDS.  Guo et al. fit T on held-out data.
   If we fitted T on a fold's own validation predictions and then scored
   that same fold, we would be fitting and evaluating on the same beats.
   Instead, for each fold k, T is fitted on the pooled validation
   predictions of the OTHER four folds and applied to fold k. Because the
   folds are patient-disjoint, T never sees a beat from a patient it is
   later applied to -- the calibration is inter-patient as well.

3. THE REJECT CURVE.  Chow, "On optimum recognition error and reject
   tradeoff" (IEEE Trans. Inf. Theory, 1970), is the origin of the reject
   option: reject when confidence falls below a threshold, and that
   threshold IS the cost you assign to a rejection. So you choose a cost
   ratio, not a reject rate.

   We do not choose either. This script sweeps the whole range and
   reports the Accuracy-Reject Curve: rejection rate on the x-axis,
   quality on the ACCEPTED beats on the y-axis. The reader picks the
   operating point. Nothing arbitrary is left to defend.

   Chow's rule assumes true posteriors and, as the reject-option survey
   notes, "is not suitable when posterior probabilities are estimated".
   That is the reason step 1 comes before step 3 and not after.

4. CLASS S PRECISION-RECALL.  Same principle applied to the question of
   what to optimise a threshold for: do not pick a precision floor, plot
   the curve. The score is the calibrated P(S).

Multiple seeds
--------------
The script uses every artifacts/preds* directory it finds. T is fitted
separately per seed, every curve is computed per seed, and the figures
show the mean with a band across seeds. A curve whose band is wider than
the effect you want to claim is not evidence, and the band is the only
honest way to show that.

Outputs, all under artifacts/stage3/
    reliability.png    calibration before and after, with ECE
    arc.png            the Accuracy-Reject Curve
    pr_class_s.png     Class S precision-recall
    stage3_results.json   every number in the figures
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

# np.trapz was renamed np.trapezoid in NumPy 2.0. Support both so the
# script does not depend on which NumPy the runner happens to have.
_trapz = getattr(np, 'trapezoid', None) or np.trapz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

N_FOLDS = 5
ECE_BINS = 15          # Guo et al. use 15 equally-spaced bins
PROB_FLOOR = 1e-12     # guards log(0) from float32 underflow
MAX_REJECT = 0.50      # sweep 0-50% rejection; beyond that the minority
                       # classes are gone and macro-F1 is meaningless
MIN_CLASS_ON_CURVE = 20   # flag any point where an eval class falls below
MIN_BIN_FOR_PLOT = 50     # reliability bins thinner than this are not drawn
MIN_FOLD_ACCEPTED = 50    # a fold holding fewer accepted beats ends the sweep


# --------------------------------------------------------------- loading

def pred_dirs(requested=None):
    """Every seed's prediction directory, as {seed: path}.

    Seed 0 lives in artifacts/preds/ (the original flat path); other seeds
    in artifacts/preds_seed{n}/. That layout is set by run_grid.pred_dir().
    """
    found = {}
    base = os.path.join(C.OUT_DIR, 'preds')
    if os.path.isdir(base):
        found[0] = base
    for p in sorted(glob.glob(os.path.join(C.OUT_DIR, 'preds_seed*'))):
        try:
            found[int(os.path.basename(p).replace('preds_seed', ''))] = p
        except ValueError:
            continue
    if requested is not None:
        missing = [s for s in requested if s not in found]
        if missing:
            raise SystemExit(f'no prediction directory for seed(s) {missing}')
        found = {s: found[s] for s in requested}
    if not found:
        raise SystemExit(
            f'No predictions found under {C.OUT_DIR}.\n'
            '  Expected artifacts/preds/ (and optionally preds_seed1..4).\n'
            '  These come from run_grid.py; this script cannot create them.')
    return found


def load_config(directory, model, strategy, use_rr):
    """Pool the five folds of one configuration from one seed directory.

    Returns (proba, y_true, fold_id). fold_id is kept because T is fitted
    leave-one-fold-out and we must know which rows belong to which fold.
    """
    rr = 'rr' if use_rr else 'norr'
    P, Y, F = [], [], []
    for f in range(N_FOLDS):
        path = os.path.join(directory, f'{model}_{strategy}_{rr}_f{f}.npz')
        if not os.path.exists(path):
            raise SystemExit(
                f'missing: {path}\n'
                f'  Need all {N_FOLDS} folds of {model}/{strategy}/{rr}.\n'
                '  Ask Rishabh for the five files for this configuration.')
        d = np.load(path)
        P.append(d['proba'].astype(np.float64))
        Y.append(d['y_true'])
        F.append(np.full(len(d['y_true']), f))
    return np.vstack(P), np.concatenate(Y), np.concatenate(F)


# ----------------------------------------------------- temperature scaling

def to_logits(p):
    """log(p), which equals the logits up to a per-row additive constant.

    float32 probabilities underflow to exactly 0 for classes the model is
    certain about, so a floor is needed. The floor only ever raises a
    probability that was already astronomically small, and the count is
    printed so the effect is visible rather than assumed.
    """
    n_zero = int((p <= 0).sum())
    return np.log(np.clip(p, PROB_FLOOR, None)), n_zero


def temper(z, T):
    """softmax(z / T), computed stably."""
    zt = z / T
    return np.exp(zt - logsumexp(zt, axis=1, keepdims=True))


def nll(z, y, T):
    zt = z / T
    logq = zt - logsumexp(zt, axis=1, keepdims=True)
    return float(-logq[np.arange(len(y)), y].mean())


def fit_temperature(z, y):
    """Fit T by minimising NLL, exactly as Guo et al. specify.

    Q is excluded. Q carries no weight in training and no weight in the
    metric (8 beats in all of DS1), so letting it steer the calibration
    would be inconsistent with every other stage.
    """
    keep = np.isin(y, C.EVAL_LABELS)
    z, y = z[keep], y[keep]
    res = minimize_scalar(lambda t: nll(z, y, t), bounds=(0.05, 10.0),
                          method='bounded')
    return float(res.x)


def calibrate_cross_fold(proba, y, fold):
    """Temperature-scale every fold with a T fitted on the other folds.

    Returns (calibrated proba, {fold: T}).
    """
    z, n_zero = to_logits(proba)
    out = np.empty_like(z)
    Ts = {}
    for f in range(N_FOLDS):
        held = fold == f
        Ts[f] = fit_temperature(z[~held], y[~held])
        out[held] = temper(z[held], Ts[f])
    return out, Ts, n_zero


# -------------------------------------------------------------- metrics

def ece(conf, correct, bins=ECE_BINS):
    """Expected Calibration Error: equally-spaced bins, weighted average
    of |accuracy - confidence| per bin (Guo et al., section 2)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            curve.append((float((lo + hi) / 2), None, None, 0))
            continue
        acc, cf = float(correct[m].mean()), float(conf[m].mean())
        total += m.mean() * abs(acc - cf)
        curve.append((float((lo + hi) / 2), acc, cf, int(m.sum())))
    return float(total), curve


def mean_reliability(per_seed, key):
    """Average each confidence bin across seeds, skipping empty bins.

    'n' is the TOTAL number of beats in the bin across seeds, not a mean.
    An earlier version averaged it, which printed n = 0 beside a real
    accuracy whenever a single seed contributed a single beat -- and a bin
    holding one beat was then drawn in the reliability diagram as a point
    at accuracy 0.0 or 1.0, which is what produced the spurious hook at
    the left-hand end of the curve. 'reliable' marks the bins with enough
    beats to plot; make_figures() draws only those.
    """
    n_bins = len(per_seed[0][key])
    out = []
    for b in range(n_bins):
        accs = [s[key][b][1] for s in per_seed if s[key][b][1] is not None]
        cfs = [s[key][b][2] for s in per_seed if s[key][b][2] is not None]
        total = int(sum(s[key][b][3] for s in per_seed))
        out.append({
            'bin_centre': per_seed[0][key][b][0],
            'confidence': float(np.mean(cfs)) if cfs else None,
            'accuracy': float(np.mean(accs)) if accs else None,
            'n': total,
            'reliable': bool(total >= MIN_BIN_FOR_PLOT),
        })
    return out


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average='macro',
                          labels=C.EVAL_LABELS, zero_division=0))


def class_f1(y_true, y_pred, cls):
    i = C.CLASS_TO_INT[cls]
    return float(f1_score(y_true, y_pred, average=None,
                          labels=list(range(len(C.CLASS_NAMES))),
                          zero_division=0)[i])


def accuracy_reject_curve(proba, y, fold, n_points=51):
    """Sweep the rejection rate and measure quality on what is accepted.

    Rejection is by coverage rather than by a threshold value, so the
    x-axis is evenly spaced and comparable across seeds. The confidence
    threshold that produces each point is recorded too, since that is the
    quantity Chow's rule talks about. Beats are ranked by max calibrated
    probability -- the confidence measure in Chow's formulation.

    THE SCORE IS THE MEAN OVER THE FIVE FOLDS, not a score on the pooled
    beats, and each fold rejects its own least-confident fraction. That
    matters more than it sounds. run_grid.py reports macro-F1 as a mean
    over folds, and on this dataset the two reductions have very different
    stability: across five seeds the pooled macro-F1 moves by ±0.0325
    while the mean over folds moves by ±0.0087, nearly four times less.
    Pooling mixes folds whose class balance is wildly different -- record
    208 holds 90% of DS1's F beats, so one fold dominates the pooled F
    score and drags the whole curve around with it.

    Using the pooled score here would have buried the actual result under
    a noise band four times wider than it needed to be, and would also
    have made this curve incomparable with every number in Stage 2.
    """
    conf = proba.max(1)
    pred = proba.argmax(1)
    n = len(y)
    fold_idx = [np.where(fold == f)[0] for f in range(N_FOLDS)]
    fold_order = [idx[np.argsort(-conf[idx], kind='stable')]
                  for idx in fold_idx]

    pts = []
    for frac in np.linspace(0.0, MAX_REJECT, n_points):
        keeps = [int(round(len(idx) * (1.0 - frac))) for idx in fold_idx]
        if min(keeps) < MIN_FOLD_ACCEPTED:
            break
        sel = [o[:k] for o, k in zip(fold_order, keeps)]
        # Counts are pooled even though the scores are not: the thin flag
        # is about whether the number we would QUOTE rests on enough
        # beats, and a pooled count is what a reader would check.
        pooled = np.concatenate(sel)
        counts = {c: int((y[pooled] == C.CLASS_TO_INT[c]).sum())
                  for c in C.EVAL_CLASSES}
        # Each metric uses the SAME reduction run_grid.py uses for it, so
        # the 0% point of this curve reproduces the Stage 2 numbers
        # exactly. run_grid.py is not uniform about this and neither can
        # we be: macro_f1_mean is a mean over folds, per_class_f1 is
        # computed on the pooled beats.
        #
        # For Class S the choice is not cosmetic, it reverses the result.
        # Record 209 holds 383 of DS1's 943 S beats, so four of the five
        # folds carry very few S beats; their per-fold S F1 is fragile and
        # collapses as soon as beats are rejected, which drags a mean over
        # folds downwards even while the pooled score rises. Both are
        # reported. The pooled one is the headline because it is what
        # Stage 2 quotes, and the disagreement between them is itself a
        # finding about the database rather than about the method.
        pts.append({
            'reject_rate': float(frac),
            'coverage': float(len(pooled) / n),
            'threshold': float(np.mean([conf[s[-1]] for s in sel])),
            'macro_f1': float(np.mean([macro_f1(y[s], pred[s]) for s in sel])),
            'f1_S': class_f1(y[pooled], pred[pooled], 'S'),
            'f1_S_byfold': float(np.mean([class_f1(y[s], pred[s], 'S')
                                          for s in sel])),
            'accepted_counts': counts,
            'thin': min(counts.values()) < MIN_CLASS_ON_CURVE,
        })
    return pts


def au_arc(pts, key='macro_f1'):
    """Area under the ARC over the swept coverage range, normalised to
    that range so it reads on the same scale as the metric itself.

    Reported with its range because AURC values are only comparable when
    the sweep is identical -- there is no single conventional range.
    """
    cov = np.array([p['coverage'] for p in pts])
    val = np.array([p[key] for p in pts])
    o = np.argsort(cov)
    cov, val = cov[o], val[o]
    width = cov[-1] - cov[0]
    return float(_trapz(val, cov) / width) if width > 0 else float(val[0])


# -------------------------------------------------------------- plotting

def band(ax, xs, series, label, color):
    """Mean line plus min-max band across seeds."""
    A = np.array(series)
    m = A.mean(0)
    ax.plot(xs, m, color=color, lw=2, label=label)
    if len(A) > 1:
        ax.fill_between(xs, A.min(0), A.max(0), color=color, alpha=0.18,
                        lw=0)
    return m


def make_figures(res, outdir, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seeds = res['seeds']
    ns = len(seeds)
    band_note = (f'mean of {ns} seeds, band = min-max' if ns > 1
                 else 'single seed - no noise band')

    # ---- 1. reliability -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 5), layout='constrained')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='perfect calibration')
    for key, colour, lab in (('before', '#c0392b', 'uncalibrated'),
                             ('after', '#1f77b4', 'temperature-scaled')):
        xs, ys = [], []
        for pt in res['reliability'][key]:
            # Bins holding a handful of beats give an accuracy of exactly
            # 0 or 1 and produce a hook at the end of the curve that is
            # noise, not miscalibration. They stay in the JSON and in ECE
            # (where they are weighted by their count and contribute
            # almost nothing); they are only kept out of the picture.
            if pt['accuracy'] is not None and pt['reliable']:
                xs.append(pt['confidence'])
                ys.append(pt['accuracy'])
        ax.plot(xs, ys, 'o-', color=colour, ms=4, lw=1.6,
                label=f"{lab}  (ECE {res['ece'][key]:.4f})")
    ax.set_xlabel('mean predicted confidence')
    ax.set_ylabel('observed accuracy')
    ax.set_title(f'Reliability — {title}', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(outdir, 'reliability.png'), dpi=150)
    plt.close(fig)

    # ---- 2. ARC ---------------------------------------------------------
    xs = res['arc']['reject_rate']
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')
    band(ax, xs, res['arc']['macro_f1'], 'macro-F1 (N,S,V,F)', '#1f77b4')
    band(ax, xs, res['arc']['f1_S'], 'Class S F1', '#e67e22')
    thin = res['arc']['thin_from']
    if thin is not None:
        ax.axvspan(thin, max(xs), color='grey', alpha=0.13, lw=0)
        # x in data coordinates, y as a fraction of the axes, so the label
        # sits inside the plot whatever the score range turns out to be.
        ax.text(thin + 0.005, 0.04,
                f'fewer than {MIN_CLASS_ON_CURVE} beats\nof some class accepted',
                transform=ax.get_xaxis_transform(),
                fontsize=7, color='#555', va='bottom', clip_on=True)
    ax.set_xlabel('rejection rate')
    ax.set_ylabel('score on accepted beats')
    ax.set_title(f'Accuracy-Reject Curve — {title}\n{band_note}', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.25)
    ax.set_xlim(0, max(xs))
    fig.savefig(os.path.join(outdir, 'arc.png'), dpi=150)
    plt.close(fig)

    # ---- 3. Class S PR --------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 4.6), layout='constrained')
    grid = res['pr_S']['recall_grid']
    band(ax, grid, res['pr_S']['precision'], 'Class S', '#e67e22')
    base = res['pr_S']['prevalence']
    ax.axhline(base, color='k', ls='--', lw=1,
               label=f'prevalence ({base:.3f})')
    aps = res['pr_S']['average_precision']
    ax.set_xlabel('recall (sensitivity)')
    ax.set_ylabel('precision (positive predictivity)')
    ax.set_title(f'Class S precision-recall — {title}\n'
                 f'AP {np.mean(aps):.4f}'
                 + (f' ± {(max(aps) - min(aps)) / 2:.4f}' if ns > 1 else ''),
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(os.path.join(outdir, 'pr_class_s.png'), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='cnn', choices=['cnn', 'xgb', 'lr'])
    ap.add_argument('--strategy', default='focal')
    ap.add_argument('--rr', type=int, default=1,
                    help='1 = with RR features (the real configuration)')
    ap.add_argument('--seeds', nargs='+', type=int, default=None,
                    help='default: every preds directory found')
    a = ap.parse_args()

    use_rr = bool(a.rr)
    title = f'{a.model} / {a.strategy} / {"rr" if use_rr else "no rr"}'
    dirs = pred_dirs(a.seeds)
    outdir = os.path.join(C.OUT_DIR, 'stage3')
    os.makedirs(outdir, exist_ok=True)

    print('=' * 72)
    print(f'STAGE 3 — DECISION LAYER   ({title})')
    print('=' * 72)
    print(f'seeds found : {sorted(dirs)}')
    print(f'reading     : {C.OUT_DIR}')
    print(f'writing     : {outdir}\n')

    per_seed = []
    for seed in sorted(dirs):
        proba, y, fold = load_config(dirs[seed], a.model, a.strategy, use_rr)
        cal, Ts, n_zero = calibrate_cross_fold(proba, y, fold)

        keep = np.isin(y, C.EVAL_LABELS)      # Q out, as everywhere else
        pb, pc, yy = proba[keep], cal[keep], y[keep]

        e_before, rel_before = ece(pb.max(1), pb.argmax(1) == yy)
        e_after, rel_after = ece(pc.max(1), pc.argmax(1) == yy)

        # Guo et al.: T must not change the prediction. Verified, not
        # assumed -- if this ever trips, the calibration is wrong.
        moved = int((pb.argmax(1) != pc.argmax(1)).sum())
        assert moved == 0, (
            f'temperature scaling changed {moved} predictions; it cannot. '
            'Check to_logits()/temper().')

        arc = accuracy_reject_curve(pc, yy, fold[keep])

        # At 0% rejection nothing has been rejected, so the curve must
        # start exactly on the Stage 2 numbers. An earlier version scored
        # Class S as a mean over folds while run_grid.py pools it; the
        # curve then began at 0.438 while the cross-check line above it
        # printed 0.407, and the reject-option result came out with the
        # opposite sign. Nothing errored. This is that check.
        base_macro = float(np.mean([
            macro_f1(yy[fold[keep] == f], pc.argmax(1)[fold[keep] == f])
            for f in range(N_FOLDS)]))
        base_s = class_f1(yy, pc.argmax(1), 'S')
        for name, got, want in (('macro_f1', arc[0]['macro_f1'], base_macro),
                                ('f1_S', arc[0]['f1_S'], base_s)):
            assert abs(got - want) < 1e-9, (
                f'seed {seed}: the reject curve starts at {name}={got:.4f} '
                f'but the Stage 2 reduction gives {want:.4f}. The curve and '
                'the cross-check are not computing the same thing.')

        s_idx = C.CLASS_TO_INT['S']
        prec, rec, _ = precision_recall_curve((yy == s_idx).astype(int),
                                              pc[:, s_idx])
        grid = np.linspace(0.0, 1.0, 101)
        # interpolate onto a common recall grid so seeds can be averaged;
        # recall from precision_recall_curve is descending.
        prec_on_grid = np.interp(grid, rec[::-1], prec[::-1])

        per_seed.append({
            'seed': seed, 'T': Ts, 'T_mean': float(np.mean(list(Ts.values()))),
            'n_beats': int(keep.sum()), 'n_underflow': n_zero,
            'ece_before': e_before, 'ece_after': e_after,
            'rel_before': rel_before, 'rel_after': rel_after,
            'arc': arc,
            'pr_prec': prec_on_grid.tolist(),
            'ap': float(average_precision_score((yy == s_idx).astype(int),
                                                pc[:, s_idx])),
            'prevalence': float((yy == s_idx).mean()),
            'macro_f1_full': macro_f1(yy, pc.argmax(1)),
            'f1_S_full': class_f1(yy, pc.argmax(1), 'S'),
            # run_grid.py reports macro-F1 as the MEAN OF THE FIVE FOLDS,
            # while everything on the reject curve is computed on the
            # pooled beats. The two are not the same number. Both are
            # printed so that "this doesn't match Stage 2" never costs
            # anyone an afternoon: the per-fold mean is what should match.
            'macro_f1_byfold': float(np.mean([
                macro_f1(yy[fold[keep] == f], pc.argmax(1)[fold[keep] == f])
                for f in range(N_FOLDS)])),
        })

        print(f'seed {seed}:  T = '
              + ', '.join(f'{Ts[f]:.3f}' for f in range(N_FOLDS))
              + f'   ECE {e_before:.4f} -> {e_after:.4f}'
              + f'   (argmax unchanged, {n_zero:,} underflows floored)')

    # ---------------------------------------------------- assemble + report
    n_pts = min(len(s['arc']) for s in per_seed)
    xs = [per_seed[0]['arc'][i]['reject_rate'] for i in range(n_pts)]
    thin_idx = [i for i in range(n_pts)
                if any(s['arc'][i]['thin'] for s in per_seed)]

    res = {
        'configuration': {'model': a.model, 'strategy': a.strategy,
                          'use_rr': use_rr},
        'seeds': [s['seed'] for s in per_seed],
        'temperature': {str(s['seed']): s['T_mean'] for s in per_seed},
        'ece': {'before': float(np.mean([s['ece_before'] for s in per_seed])),
                'after': float(np.mean([s['ece_after'] for s in per_seed]))},
        'reliability': {k: mean_reliability(per_seed, 'rel_' + k)
                        for k in ('before', 'after')},
        'arc': {
            'reject_rate': xs,
            'threshold': [[s['arc'][i]['threshold'] for i in range(n_pts)]
                          for s in per_seed],
            'macro_f1': [[s['arc'][i]['macro_f1'] for i in range(n_pts)]
                         for s in per_seed],
            'f1_S': [[s['arc'][i]['f1_S'] for i in range(n_pts)]
                     for s in per_seed],
            'f1_S_byfold': [[s['arc'][i]['f1_S_byfold'] for i in range(n_pts)]
                            for s in per_seed],
            'thin_from': xs[thin_idx[0]] if thin_idx else None,
            'au_arc_macro_f1': [au_arc(s['arc'][:n_pts]) for s in per_seed],
            'au_arc_f1_S': [au_arc(s['arc'][:n_pts], 'f1_S')
                            for s in per_seed],
            'au_arc_coverage_range': [1.0 - max(xs), 1.0],
        },
        'pr_S': {
            'recall_grid': np.linspace(0, 1, 101).tolist(),
            'precision': [s['pr_prec'] for s in per_seed],
            'average_precision': [s['ap'] for s in per_seed],
            'prevalence': float(np.mean([s['prevalence'] for s in per_seed]))},
        'no_rejection': {
            'macro_f1_pooled': [s['macro_f1_full'] for s in per_seed],
            'macro_f1_mean_of_folds': [s['macro_f1_byfold']
                                       for s in per_seed],
            'f1_S': [s['f1_S_full'] for s in per_seed]},
    }

    def spread(v):
        return f'{np.mean(v):.4f}' + (f' ± {(max(v) - min(v)) / 2:.4f}'
                                      if len(v) > 1 else '')

    T_mean = float(np.mean([s['T_mean'] for s in per_seed]))
    direction = ('the model was OVERconfident and is being softened'
                 if T_mean > 1 else
                 'the model was UNDERconfident and is being sharpened')
    print(f"\ntemperature   {spread([s['T_mean'] for s in per_seed])}"
          f'\n              T {">" if T_mean > 1 else "<"} 1: {direction}.')
    if T_mean < 1:
        # Guo et al. found modern networks overconfident (T > 1), but they
        # studied cross-entropy. Mukhoti et al. (NeurIPS 2020) show focal
        # loss raises the entropy of the predicted distribution, "thereby
        # preventing the model from becoming overconfident" -- so T < 1 for
        # a focal-trained model is the documented direction, not a fault.
        print('              Expected for focal loss — see Mukhoti et al. '
              '(2020).')
    print(f"ECE           {res['ece']['before']:.4f} -> "
          f"{res['ece']['after']:.4f}")
    print('\nat 0% rejection — calibration cannot move these, so they are')
    print('the Stage 2 numbers and are printed as a cross-check:')
    print(f"  macro-F1, mean of folds  "
          f"{spread(res['no_rejection']['macro_f1_mean_of_folds'])}"
          '   <- compare with grid_summary.json')
    print(f"  Class S F1 (pooled)      "
          f"{spread(res['no_rejection']['f1_S'])}"
          '   <- compare with grid_summary.json')
    print(f"  macro-F1, pooled         "
          f"{spread(res['no_rejection']['macro_f1_pooled'])}"
          '   <- the left end of the reject curve')

    print('\nALONG THE REJECT CURVE')
    print(f"{'reject':>7} {'thresh':>8} {'macro-F1':>18} {'Class S F1':>18}")
    print('-' * 56)
    for i in range(n_pts):
        if abs(xs[i] * 100 - round(xs[i] * 100)) > 1e-9 or \
                round(xs[i] * 100) % 5 != 0:
            continue
        mark = '  (thin)' if i in thin_idx else ''
        print(f'{xs[i] * 100:6.0f}% '
              f"{np.mean([s['arc'][i]['threshold'] for s in per_seed]):8.3f} "
              f"{spread([s['arc'][i]['macro_f1'] for s in per_seed]):>18} "
              f"{spread([s['arc'][i]['f1_S'] for s in per_seed]):>18}{mark}")

    # The table above is the WRONG test for "does rejecting help?" and it
    # is easy to read it as the right one. Its ± is the spread of the
    # SCORE between seeds -- different random initialisations land at
    # different places. The question is whether rejecting improves a
    # given model, which is a PAIRED comparison: each seed against its
    # own 0% point. The paired gain is far smaller than the between-seed
    # spread, so a gain that looks lost in the noise above can still be
    # present in every single run. Counting how many of the seeds move
    # the same way is the statistic that answers the question.
    if len(per_seed) > 1:
        print('\nPAIRED GAIN — each seed against ITS OWN 0% point')
        print('(the between-seed spread above is not the right '
              'comparison for this)')
        print(f"\n{'reject':>7} {'macro-F1':>22} {'Class S (pooled)':>22}"
              f" {'Class S (by fold)':>22}")
        print('-' * 76)
        for i in range(n_pts):
            if abs(xs[i] * 100 - round(xs[i] * 100)) > 1e-9 or \
                    round(xs[i] * 100) % 5 != 0 or i == 0:
                continue
            row = ''
            for k in ('macro_f1', 'f1_S', 'f1_S_byfold'):
                g = [s['arc'][i][k] - s['arc'][0][k] for s in per_seed]
                up = sum(x > 0 for x in g)
                row += f'   {np.mean(g):+.4f}  {up}/{len(g)} up'.rjust(22)
            mark = '  (thin)' if i in thin_idx else ''
            print(f'{xs[i] * 100:6.0f}% {row}{mark}')
        print('\nA gain present in every seed is a real effect however '
              'small;\na larger mean gain that only half the seeds show '
              'is not.')

    lo = 1.0 - max(xs)
    print(f"\nAU-ARC (coverage {lo:.2f}-1.00)")
    print(f"  macro-F1    {spread(res['arc']['au_arc_macro_f1'])}")
    print(f"  Class S F1  {spread(res['arc']['au_arc_f1_S'])}")
    print(f"\nClass S average precision  "
          f"{spread(res['pr_S']['average_precision'])}"
          f"   (prevalence {res['pr_S']['prevalence']:.4f})")
    if thin_idx:
        print(f'\nNote: beyond {xs[thin_idx[0]] * 100:.0f}% rejection some '
              f'evaluated class has fewer than {MIN_CLASS_ON_CURVE} accepted '
              'beats.\n      Those points are shaded in the figure and '
              'should not be quoted.')

    with open(os.path.join(outdir, 'stage3_results.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    make_figures(res, outdir, title)

    print(f'\nWrote stage3_results.json, reliability.png, arc.png, '
          f'pr_class_s.png\n  -> {outdir}')
    print('DS2 was not opened.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
