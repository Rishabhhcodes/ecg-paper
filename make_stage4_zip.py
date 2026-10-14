"""
Pack the Stage 2 predictions Stage 4 needs, and nothing else.

Run on the laptop, from the repository root (C:\\ecg-paper):

    python make_stage4_zip.py

Writes stage4_preds.zip — 50 files, a few MB.

This zip is OPTIONAL. Stage 4 runs without it and says so in its output;
all it adds is one secondary result, whether a temperature fitted on DS1
still calibrates predictions on unseen patients. Send it if it is ready,
but do not hold up the evaluation for it.

Only the two CNN configurations Stage 4 reports are included. The runner
rebuilds ds1.npz and ds2.npz themselves from PhysioNet, so no dataset
needs sending — and nothing here can contain anything from DS2, since
these files were written before this stage existed.
"""
import os
import sys
import zipfile

CONFIGS = ['cnn_class_weight_rr', 'cnn_focal_rr']
SEED_DIRS = ['preds', 'preds_seed1', 'preds_seed2', 'preds_seed3',
             'preds_seed4']
N_FOLDS = 5
OUT = 'stage4_preds.zip'

if not os.path.isdir('artifacts'):
    sys.exit('Run this from the repository root — no artifacts/ folder here.\n'
             '  cd /c/ecg-paper   (Git Bash)')

found, missing = [], []
for d in SEED_DIRS:
    for cfg in CONFIGS:
        for f in range(N_FOLDS):
            p = os.path.join('artifacts', d, f'{cfg}_f{f}.npz')
            (found if os.path.exists(p) else missing).append(p)

print(f'found   {len(found)} of {len(found) + len(missing)} files')
if missing:
    print(f'MISSING {len(missing)}:')
    for p in missing[:12]:
        print('   ', p)
    if len(missing) > 12:
        print(f'    ... and {len(missing) - 12} more')
    sys.exit('\nIncomplete. Either unzip the multi-seed download into\n'
             'artifacts/ first, or send nothing — Stage 4 runs without this\n'
             'zip and simply omits the calibration-transfer result. An\n'
             'incomplete set is worse than none, because the temperature\n'
             'would be fitted on fewer folds than it reports.')

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in found:
        z.write(p, p.replace(os.sep, '/'))

print(f'\nwrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, '
      f'{len(found)} files)')
print('Paths inside start with artifacts/, so it unzips at the repo root.')