"""Stage-1 labels: convert the RCSB clusters' dynamic_contacts.csv to .pt, checking each one.

    python investigation/scripts/prepare_rcsb_labels.py investigation/data/rcsb/rcsb/rcsb \
        --workers 16 --report investigation/results/15_rcsb_labels.csv

Each `dynamic_contacts.csv` holds one 0/1 integer per line: the strict upper triangle of the
L x L dynamic-contact matrix (L(L-1)/2 entries, not L^2), in row-major order
(`np.triu_indices(L, 1)`). Row-major is inferred from the data: on 300 random clusters it puts
2 of 12,767 positives at |i - j| = 1 against 103 for column-major (adjacent residues cannot be a
dynamic contact).

Does what the authors' `decompress_rcsb_dataset.py` does (writes `dynamic_contacts.pt` next to
each csv, the flat vector as `torch.int`), and additionally records per cluster: L, number of
entries, number of positives, and whether the count equals L(L-1)/2. Clusters whose count does
not match are listed in the report and excluded downstream.
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch


def convert(folder):
    folder = Path(folder)
    seq = (folder / "consensus.fasta").read_text().splitlines()[1].strip()
    L = len(seq)
    y = np.ravel(np.loadtxt(folder / "dynamic_contacts.csv", dtype=np.int32))
    torch.save(torch.tensor(y, dtype=torch.int), folder / "dynamic_contacts.pt")
    ok = y.size == L * (L - 1) // 2 and set(np.unique(y).tolist()) <= {0, 1}
    return folder.name, L, int(y.size), int((y == 1).sum()), int(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    folders = sorted(p.parent for p in Path(args.root).glob("*/dynamic_contacts.csv"))
    print(f"{len(folders)} clusters", flush=True)
    with Pool(args.workers) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(convert, folders, chunksize=64), 1):
            rows.append(r)
            if k % 10000 == 0:
                print(f"  {k}", flush=True)
    rows.sort()
    with open(args.report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "L", "n_entries", "n_positive", "ok"])
        w.writerows(rows)
    bad = [r for r in rows if not r[4]]
    print(f"done: {len(rows)} converted, {len(bad)} with a count other than L(L-1)/2")
    for r in bad[:50]:
        print("  BAD", r)


if __name__ == "__main__":
    main()
