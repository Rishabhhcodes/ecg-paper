"""
Boxes 3-6 — beat segmentation, AAMI mapping, per-record normalisation,
RR-interval feature extraction.

Design rules enforced here:
  * R-peak locations come from the database's expert annotations, never
    from a detector. Detector error is a separate variable we do not want
    confounding the imbalance results.
  * Normalisation is per-record, so no statistic ever crosses a patient
    boundary. This is what makes it leak-free by construction.
  * patient_id (the record number) is attached to every single beat and
    must survive to the very end. GroupKFold in Stage 2 depends on it.
"""
import os
import sys

import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from config import (AAMI_MAP, CLASS_TO_INT, LEAD, LOCAL_RR_WINDOW,
                    POST_SAMPLES, PRE_SAMPLES, SECOND_LEAD_FALLBACK, FS)


def load_record(record_id, raw_dir):
    """Read one record's signal and beat annotations."""
    path = os.path.join(raw_dir, str(record_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, 'atr')

    # Prefer lead MLII; fall back to channel 0 if this record lacks it.
    if LEAD in rec.sig_name:
        ch = rec.sig_name.index(LEAD)
        lead_used = LEAD
    else:
        ch = SECOND_LEAD_FALLBACK
        lead_used = rec.sig_name[ch]

    signal = rec.p_signal[:, ch].astype(np.float64)
    return signal, ann.sample, np.array(ann.symbol), lead_used, rec.fs


def normalise_per_record(signal):
    """Box 5 — z-score using this record's own statistics.

    Removes the patient's baseline offset and amplitude scale, which is
    the cheapest defence against inter-patient variation. Because the
    statistics are computed inside one record, this cannot leak.
    """
    mu = signal.mean()
    sd = signal.std()
    if sd < 1e-8:
        sd = 1.0
    return (signal - mu) / sd


def rr_features(r_locations, fs, idx):
    """Box 6 — four RR-interval features for beat `idx`.

    Class S is defined by prematurity, not morphology. A window cropped
    around a beat's own R-peak discards exactly that information, so
    without these features S is unidentifiable by construction.

      pre_rr        seconds since previous R-peak
      post_rr       seconds until next R-peak
      local_ratio   pre_rr / mean of last LOCAL_RR_WINDOW intervals
      global_ratio  pre_rr / mean interval across the whole record

    The two ratios are relative to the patient's own rhythm, so they also
    help across patients, not just within one.
    """
    rr = np.diff(r_locations) / fs           # interval i = R[i+1] - R[i]

    pre_rr = rr[idx - 1]
    post_rr = rr[idx]

    lo = max(0, idx - 1 - LOCAL_RR_WINDOW)
    local_mean = rr[lo:idx - 1].mean() if idx - 1 > lo else pre_rr
    global_mean = rr.mean()

    local_ratio = pre_rr / local_mean if local_mean > 1e-6 else 1.0
    global_ratio = pre_rr / global_mean if global_mean > 1e-6 else 1.0

    return np.array([pre_rr, post_rr, local_ratio, global_ratio])


def segment_record(record_id, raw_dir):
    """Boxes 3-6 for a single record.

    Returns (beats, rr, labels, patient_ids, meta).
    """
    signal, ann_samples, ann_symbols, lead_used, fs = load_record(record_id, raw_dir)
    signal = normalise_per_record(signal)

    # Box 4 — keep only annotations that are beats AND map to an AAMI class.
    keep = np.array([s in AAMI_MAP for s in ann_symbols])
    r_locs = ann_samples[keep]
    symbols = ann_symbols[keep]

    n_dropped_nonbeat = int((~keep).sum())

    beats, rrs, labels = [], [], []
    n_edge_dropped = 0

    # First and last beat have no pre-RR / post-RR, so they are dropped.
    for i in range(1, len(r_locs) - 1):
        r = r_locs[i]
        start, end = r - PRE_SAMPLES, r + POST_SAMPLES

        if start < 0 or end > len(signal):
            n_edge_dropped += 1
            continue

        beats.append(signal[start:end])
        rrs.append(rr_features(r_locs, fs, i))
        labels.append(CLASS_TO_INT[AAMI_MAP[symbols[i]]])

    if not beats:
        raise RuntimeError(f'record {record_id}: no usable beats')

    beats = np.stack(beats).astype(np.float32)
    rrs = np.stack(rrs).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    pids = np.full(len(labels), int(record_id), dtype=np.int64)

    meta = {
        'record': int(record_id),
        'lead': lead_used,
        'fs': int(fs),
        'n_beats': int(len(labels)),
        'n_nonbeat_annotations_dropped': n_dropped_nonbeat,
        'n_edge_beats_dropped': n_edge_dropped + 2,
    }
    return beats, rrs, labels, pids, meta
