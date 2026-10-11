"""
Box 7 — assemble DS1 / DS2 and write them to disk.

Output (artifacts/):
    ds1.npz, ds2.npz   X (n,216)  rr (n,4)  y (n,)  pid (n,)
    manifest.json      per-record counts, class distribution, config echo

The DS1/DS2 division is by record, so no patient contributes beats to
both. Everything downstream reads only these two files.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from preprocess import segment_record


def build_split(record_ids, raw_dir, name):
    Xs, Rs, Ys, Ps, metas = [], [], [], [], []

    for rid in record_ids:
        t0 = time.time()
        beats, rr, y, pid, meta = segment_record(rid, raw_dir)
        Xs.append(beats); Rs.append(rr); Ys.append(y); Ps.append(pid)
        meta['seconds'] = round(time.time() - t0, 2)
        metas.append(meta)

        dist = {C.CLASS_NAMES[k]: int((y == k).sum()) for k in range(5)}
        dist = {k: v for k, v in dist.items() if v}
        print(f'  {name} {rid}: {meta["n_beats"]:>6} beats  {dist}')

    return (np.concatenate(Xs), np.concatenate(Rs),
            np.concatenate(Ys), np.concatenate(Ps), metas)


def class_distribution(y):
    return {C.CLASS_NAMES[k]: int((y == k).sum()) for k in range(5)}


def main(raw_dir=None, out_dir=None):
    raw_dir = raw_dir or C.RAW_DIR
    out_dir = out_dir or C.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nDS1 (training) — {len(C.DS1)} records')
    X1, R1, y1, p1, m1 = build_split(C.DS1, raw_dir, 'DS1')

    print(f'\nDS2 (test, sealed) — {len(C.DS2)} records')
    X2, R2, y2, p2, m2 = build_split(C.DS2, raw_dir, 'DS2')

    # --- integrity assertions: these are the protocol, not decoration ---
    assert set(np.unique(p1)).isdisjoint(set(np.unique(p2))), \
        'PATIENT OVERLAP between DS1 and DS2 — protocol violated'
    for pid in list(C.PACED_RECORDS):
        assert pid not in p1 and pid not in p2, f'paced record {pid} leaked in'
    assert X1.shape[1] == C.WINDOW_LEN and R1.shape[1] == 4
    assert len(X1) == len(R1) == len(y1) == len(p1)

    np.savez_compressed(os.path.join(out_dir, 'ds1.npz'),
                        X=X1, rr=R1, y=y1, pid=p1)
    np.savez_compressed(os.path.join(out_dir, 'ds2.npz'),
                        X=X2, rr=R2, y=y2, pid=p2)

    manifest = {
        'protocol': 'de Chazal et al. 2004 inter-patient (DS1/DS2)',
        'standard': 'ANSI/AAMI EC57',
        'paced_records_excluded': C.PACED_RECORDS,
        'window': {'pre_ms': C.WINDOW_PRE_MS, 'post_ms': C.WINDOW_POST_MS,
                   'samples': C.WINDOW_LEN, 'fs': C.FS},
        'rr_features': ['pre_rr', 'post_rr', 'local_ratio', 'global_ratio'],
        'normalisation': 'per-record z-score',
        'ds1': {'records': C.DS1, 'n_beats': int(len(y1)),
                'class_distribution': class_distribution(y1), 'per_record': m1},
        'ds2': {'records': C.DS2, 'n_beats': int(len(y2)),
                'class_distribution': class_distribution(y2), 'per_record': m2},
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print('\n' + '=' * 62)
    print(f'{"":8}{"total":>10}' + ''.join(f'{c:>8}' for c in C.CLASS_NAMES))
    for nm, y in (('DS1', y1), ('DS2', y2)):
        d = class_distribution(y)
        print(f'{nm:8}{len(y):>10,}' + ''.join(f'{d[c]:>8,}' for c in C.CLASS_NAMES))
    print('=' * 62)

    imb = class_distribution(y1)
    print(f'DS1 imbalance ratio N:F = {imb["N"] / max(imb["F"], 1):.0f}:1')
    print(f'\nWritten to {out_dir}/  (ds1.npz, ds2.npz, manifest.json)')
    return manifest


if __name__ == '__main__':
    main(raw_dir=sys.argv[1] if len(sys.argv) > 1 else None)
