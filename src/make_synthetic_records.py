"""
TESTING ONLY — writes fake records in MIT-BIH WFDB format.

Purpose: exercise the real code path (wfdb.rdrecord / rdann, segmentation,
RR features, DS1/DS2 assembly) on a machine that cannot reach PhysioNet.
The waveforms are crude synthetic ECG, NOT real data. Never use the
resulting arrays for any experiment that goes in the paper.

Delete this file, or leave it as a unit-test fixture. It is not part of
the pipeline.
"""
import os
import sys

import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
import config as C

RNG = np.random.default_rng(0)


def _beat_template(kind, fs):
    """Very rough beat shapes: enough to test plumbing, not physiology."""
    t = np.arange(-int(0.25 * fs), int(0.45 * fs)) / fs
    p = 0.12 * np.exp(-((t + 0.16) ** 2) / (2 * 0.020 ** 2))
    tw = 0.28 * np.exp(-((t - 0.28) ** 2) / (2 * 0.055 ** 2))

    if kind == 'V':                                   # wide, no P
        qrs = 1.1 * np.exp(-(t ** 2) / (2 * 0.045 ** 2))
        return qrs + tw
    if kind == 'F':                                   # hybrid
        qrs = 1.0 * np.exp(-(t ** 2) / (2 * 0.028 ** 2))
        return 0.5 * p + qrs + tw
    qrs = 1.0 * np.exp(-(t ** 2) / (2 * 0.014 ** 2))  # narrow, normal-ish
    return p + qrs + tw


def make_record(record_id, out_dir, n_beats=900, fs=C.FS):
    """Write <record_id>.dat/.hea/.atr with two channels and beat labels."""
    sym_pool = ['N'] * 84 + ['A'] * 4 + ['V'] * 8 + ['F'] * 2 + ['Q'] * 2
    symbols, r_locs = [], []

    pos = int(0.8 * fs)
    base_rr = RNG.uniform(0.70, 0.95)
    sig = np.zeros(int(n_beats * base_rr * fs) + 4 * fs)

    for _ in range(n_beats):
        sym = sym_pool[RNG.integers(len(sym_pool))]
        aami = C.AAMI_MAP[sym]

        # S beats arrive early — the whole point of the RR features.
        rr = base_rr * (0.62 if aami == 'S' else RNG.uniform(0.95, 1.05))
        pos += int(rr * fs)
        if pos + int(0.5 * fs) >= len(sig):
            break

        tmpl = _beat_template(aami, fs) * RNG.uniform(0.9, 1.1)
        s = pos - int(0.25 * fs)
        sig[s:s + len(tmpl)] += tmpl

        r_locs.append(pos)
        symbols.append(sym)

    sig = sig[:pos + fs]
    sig = sig * RNG.uniform(0.8, 1.4) + RNG.uniform(-0.5, 0.5)   # per-patient scale/offset
    sig += 0.02 * RNG.standard_normal(len(sig))

    two_ch = np.stack([sig, sig * 0.6], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    wfdb.wrsamp(str(record_id), fs=fs, units=['mV', 'mV'],
                sig_name=['MLII', 'V1'], p_signal=two_ch,
                fmt=['212', '212'], write_dir=out_dir)

    # sprinkle non-beat annotations to test that they get filtered out
    samp = np.array(r_locs)
    syms = list(symbols)
    samp = np.append(samp, [10, len(sig) - 10])
    syms += ['+', '~']
    order = np.argsort(samp)

    wfdb.wrann(str(record_id), 'atr', samp[order],
               np.array(syms)[order], write_dir=out_dir)
    return len(r_locs)


def main(out_dir='data/synthetic'):
    total = 0
    for rid in C.ALL_RECORDS:
        total += make_record(rid, out_dir, n_beats=int(RNG.integers(700, 1100)))
    print(f'Wrote {len(C.ALL_RECORDS)} synthetic records '
          f'({total:,} beats) to {out_dir}/')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'data/synthetic')
