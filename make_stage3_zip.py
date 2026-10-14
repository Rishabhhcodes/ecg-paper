"""
Pack exactly the prediction files Stage 3 needs, and nothing else.

Run this on the laptop, from the repository root (C:\\ecg-paper):

    python make_stage3_zip.py

It writes stage3_preds.zip — about 25 files, a few MB. Send that one zip
to whoever is doing Stage 3. Nothing else from artifacts/ is needed: no
ds1.npz, no ds2.npz, no other strategy.

The script refuses to write an incomplete zip. A missing fold would make
Stage 3 silently describe four folds instead of five, which is the kind
of failure this project has already been bitten by.
"""
import glob
import os
import sys
import zipfile

CONFIG = 'cnn_focal_rr'        # the configuration Stage 3 analyses
SEED_DIRS = ['preds', 'preds_seed1', 'preds_seed2', 'preds_seed3',
             'preds_seed4']
N_FOLDS = 5
OUT = 'stage3_preds.zip'

if not os.path.isdir('artifacts'):
    sys.exit('Run this from the repository root — no artifacts/ folder here.\n'
             '  cd /c/ecg-paper   (Git Bash)')

found, missing = [], []
for d in SEED_DIRS:
    for f in range(N_FOLDS):
        p = os.path.join('artifacts', d, f'{CONFIG}_f{f}.npz')
        (found if os.path.exists(p) else missing).append(p)

print(f'found   {len(found)} files')
if missing:
    print(f'MISSING {len(missing)}:')
    for p in missing:
        print('   ', p)
    sys.exit('\nDo not send an incomplete set. If a whole seed directory is\n'
             'absent, unzip the multi-seed download into artifacts/ first.')

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in found:
        z.write(p, p.replace(os.sep, '/'))

print(f'\nwrote {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB, '
      f'{len(found)} files)')
print('Paths inside the zip start with artifacts/, so it unzips straight')
print('into the repository root.')