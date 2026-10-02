"""
Box 1 — fetch the raw MIT-BIH Arrhythmia Database from PhysioNet.

Downloads only the 44 non-paced records the protocol uses.
Roughly 90 MB. Run once; it is cached on disk afterwards.
"""
import os
import sys

import wfdb

sys.path.insert(0, os.path.dirname(__file__))
from config import ALL_RECORDS, PACED_RECORDS, RAW_DIR


def download(out_dir=RAW_DIR, include_paced=False):
    """Download MIT-BIH records.

    include_paced=True also fetches 102/104/107/217. They are never used
    for training, but having them locally lets us show in the paper that
    the exclusion was deliberate rather than accidental.
    """
    os.makedirs(out_dir, exist_ok=True)

    records = [str(r) for r in ALL_RECORDS]
    if include_paced:
        records += [str(r) for r in PACED_RECORDS]

    print(f'Downloading {len(records)} records to {out_dir}/ ...')
    wfdb.dl_database('mitdb', dl_dir=out_dir, records=records)

    missing = [r for r in records
               if not os.path.exists(os.path.join(out_dir, f'{r}.dat'))]
    if missing:
        raise RuntimeError(f'Download incomplete, missing: {missing}')

    print(f'OK — {len(records)} records present in {out_dir}/')
    return out_dir


if __name__ == '__main__':
    download(include_paced='--with-paced' in sys.argv)
