"""
Stage 1 configuration — inter-patient protocol constants.

Every number in this file traces to a citation. Do not change values
without updating the reference, because the protocol IS the contribution.
"""

# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------
# MIT-BIH Arrhythmia Database has 48 records from 47 subjects.
# ANSI/AAMI EC57 requires excluding records with paced beats.
PACED_RECORDS = [102, 104, 107, 217]

# De Chazal, O'Dwyer & Reilly (2004), IEEE TBME 51(7):1196-1206.
# DS1 = training, DS2 = test. No subject appears in both.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]

DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

ALL_RECORDS = sorted(DS1 + DS2)

# ----------------------------------------------------------------------
# AAMI class mapping
# ----------------------------------------------------------------------
# 15 MIT-BIH beat annotation symbols -> 5 AAMI EC57 classes.
# This table is the one thing to get signed off by the guide before
# any modelling happens: every downstream number depends on it.
AAMI_MAP = {
    # N — Non-ectopic
    'N': 'N',   # Normal beat
    'L': 'N',   # Left bundle branch block
    'R': 'N',   # Right bundle branch block
    'e': 'N',   # Atrial escape
    'j': 'N',   # Nodal (junctional) escape
    # S — Supraventricular ectopic
    'A': 'S',   # Atrial premature
    'a': 'S',   # Aberrated atrial premature
    'J': 'S',   # Nodal (junctional) premature
    'S': 'S',   # Supraventricular premature
    # V — Ventricular ectopic
    'V': 'V',   # Premature ventricular contraction
    'E': 'V',   # Ventricular escape
    # F — Fusion
    'F': 'F',   # Fusion of ventricular and normal
    # Q — Unknown
    '/': 'Q',   # Paced (near-absent once paced records are excluded)
    'f': 'Q',   # Fusion of paced and normal
    'Q': 'Q',   # Unclassifiable
}

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_NAMES)}

# ----------------------------------------------------------------------
# Evaluation classes  (team decision, 2026-10-01)
# ----------------------------------------------------------------------
# Q is kept as a model output but EXCLUDED from every reported metric.
#
# Why: after the AAMI-mandated removal of paced records (102/104/107/217),
# Q holds only 8 beats in DS1 and 7 in DS2 — too few to train on, to test
# on, or to appear in most CV folds. Averaging an undefined class into
# macro-F1 would depress every headline number for no scientific reason.
#
# This also matches the comparison literature. Romdhane et al. (2021),
# the two-stage RR method (2022) and ICC-WKELM (2023) all report
# N / SVEB / VEB / F, so a four-class metric is what makes Table IV a
# like-for-like comparison.
#
# Methods-section wording:
#   "Class Q was excluded from evaluation, comprising 8 and 7 beats in
#    DS1 and DS2 respectively after the AAMI-mandated removal of paced
#    records."
#
# Set to CLASS_NAMES to report all five instead.
EVAL_CLASSES = ['N', 'S', 'V', 'F']
EVAL_LABELS = [CLASS_TO_INT[c] for c in EVAL_CLASSES]   # [0, 1, 2, 3]

# Non-beat annotations (rhythm/quality markers). Ignored, never windowed.
NON_BEAT_SYMBOLS = set('[!]x()ptu`\'^|~+sT*D=@"')

# ----------------------------------------------------------------------
# Signal / windowing
# ----------------------------------------------------------------------
FS = 360                 # MIT-BIH native sampling rate (Hz)
LEAD = 'MLII'            # primary lead; falls back to channel 0 if absent
SECOND_LEAD_FALLBACK = 0

# Beat window around the R-peak.
# 200 ms before + 400 ms after keeps the preceding P wave and the
# following T wave inside the window, which Stage 5 needs for
# segment-level attribution.
WINDOW_PRE_MS = 200
WINDOW_POST_MS = 400

PRE_SAMPLES = int(FS * WINDOW_PRE_MS / 1000)    # 72
POST_SAMPLES = int(FS * WINDOW_POST_MS / 1000)  # 144
WINDOW_LEN = PRE_SAMPLES + POST_SAMPLES         # 216

# Approximate segment boundaries within the window, as sample indices.
# Used by Stage 5 to aggregate attributions into clinical regions.
# R-peak sits at index PRE_SAMPLES.
SEGMENTS = {
    'P':   (0, 54),                       # -200 ms .. -50 ms
    'QRS': (54, 108),                     #  -50 ms .. +100 ms
    'T':   (108, WINDOW_LEN),             # +100 ms .. +400 ms
}

# Local RR average is computed over this many preceding beats.
LOCAL_RR_WINDOW = 10

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
RAW_DIR = 'data/mitdb'
OUT_DIR = 'artifacts'
