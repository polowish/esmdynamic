"""Do the evaluation proteins also appear in the stage-1 (RCSB) pre-training data?

    python investigation/scripts/check_pretrain_overlap.py

The paper split RCSB clusters (train/val/test, <= 20% identity across splits) and mdCATH
domains separately; nothing says the evaluation sets were filtered against the OTHER dataset.
1. mdCATH: the shipped mdcath_to_rcsb_mapping.csv links domains to RCSB clusters; count how
   many mdCATH val/test domains map into RCSB train.
2. ATLAS test (82 chains): exact sequence matches and containment (either way) between each
   chain and the RCSB consensus sequences, and which RCSB split those clusters are in. A lower
   bound on overlap (no alignment); identity-based search would find more.
"""
import csv
import json
from collections import Counter
from pathlib import Path

D = Path("investigation/data")


def ids(p):
    return [x.strip() for x in open(p) if x.strip()]


def main():
    rsplit = {}
    for s in ("train", "val", "test"):
        for i in ids(D / "rcsb/rcsb" / f"{s}.csv"):
            rsplit[i] = s
    out = {}
    m = list(csv.DictReader(open(D / "rcsb/mdcath_to_rcsb_mapping.csv")))
    width = max(len(k) for k in rsplit)
    to_cluster = {r["mdcath_id"]: str(r["rcsb_cluster_id"]).zfill(width) for r in m}
    for s in ("train", "val", "test"):
        dom = ids(D / "mdcath" / f"{s}.csv")
        c = Counter(rsplit.get(to_cluster.get(d), "unmapped" if d not in to_cluster else "not_in_any_split")
                    for d in dom)
        out[f"mdcath_{s}"] = dict(n=len(dom), rcsb_split_of_mapped_cluster=dict(c))
    seqs = {}
    for f in (D / "rcsb/rcsb/rcsb").iterdir():
        try:
            seqs[f.name] = (f / "consensus.fasta").read_text().splitlines()[1].strip()
        except Exception:
            pass
    atlas = list(csv.DictReader(open(D / "atlas_test/atlas_test.csv")))
    by_seq = {}
    for k, s in seqs.items():
        by_seq.setdefault(s, []).append(k)
    hits = []
    for r in atlas:
        q = r["seqres"]
        exact = by_seq.get(q, [])
        contain = [k for k, s in seqs.items() if (len(q) >= 30 and q in s) or (len(s) >= 30 and s in q)] if not exact else exact
        if contain:
            hits.append(dict(name=r["name"], L=len(q), n=len(contain), exact=bool(exact),
                             splits=dict(Counter(rsplit.get(k, "no_split") for k in contain))))
    out["atlas_test"] = dict(n=len(atlas), chains_matching_rcsb_consensus=len(hits),
                             in_rcsb_train=sum("train" in h["splits"] for h in hits), hits=hits)
    print(json.dumps({k: v if k != "atlas_test" else {kk: vv for kk, vv in v.items() if kk != "hits"}
                      for k, v in out.items()}, indent=1))
    Path("investigation/results/15_pretrain_overlap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
