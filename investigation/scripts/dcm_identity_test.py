"""Check 5: removing every DCM block leaves ESMDynamic's outputs bitwise identical.

    python investigation/scripts/dcm_identity_test.py [--out investigation/results/05_dcm_identity.json]

Released V2 weights. ESMFold runs once per sequence; the three heads then run on identical
copies of its output:

  released      the model as shipped
  rerun         the same again (determinism baseline)
  no_blocks     every DCM's `blocks` replaced by an empty ModuleList, so each DCM's trunk
                applies only its positional embedding and recycling terms
  perturbed     control: one block output projection set to a tiny nonzero value, to show
                the comparison would catch a block that computes anything

Every floating-point output of every head is compared with torch.equal and max |diff|.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

SEQS = {
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "GB1": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
    "HEWL": ("KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRN"
             "LCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"),
}


def head_outputs(heads, base):
    st = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
    with torch.no_grad():
        for h in heads.values():
            st = h(st, num_recycles=None)
    return {k: v for k, v in st.items() if torch.is_tensor(v) and k not in base}


def compare(a, b):
    rows = {}
    for k in a:
        x, y = a[k], b[k]
        if x.is_floating_point():
            rows[k] = {"equal": bool(torch.equal(x, y)),
                       "max_abs_diff": float((x.float() - y.float()).abs().max())}
        else:
            rows[k] = {"equal": bool(torch.equal(x, y)), "max_abs_diff": None}
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="investigation/results/05_dcm_identity.json")
    args = p.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences

    model = esmdynamic().cuda().eval()
    released = model.heads
    no_blocks = copy.deepcopy(released)
    for h in no_blocks.values():
        h.dynamic_module.blocks = nn.ModuleList()
    perturbed = copy.deepcopy(released)
    with torch.no_grad():
        perturbed["dynamic"].dynamic_module.blocks[0].mlp_pair.mlp[-2].weight.fill_(1e-3)

    out = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "proteins": {}}
    for name, seq in SEQS.items():
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
        with torch.no_grad():
            base = model.esmfold(aatype, mask, residx, None, None)
        base["mask"] = mask
        ref = head_outputs(released, base)
        res = {"rerun": compare(ref, head_outputs(released, base)),
               "no_blocks": compare(ref, head_outputs(no_blocks, base)),
               "perturbed": compare(ref, head_outputs(perturbed, base))}
        out["proteins"][name] = {"length": len(seq), **res}
        print(f"=== {name} (L={len(seq)}): {len(ref)} head outputs compared")
        for variant, rows in res.items():
            n_eq = sum(r["equal"] for r in rows.values())
            worst = max((r["max_abs_diff"] or 0.0) for r in rows.values())
            print(f"   {variant:<10} bitwise identical: {n_eq}/{len(rows)}   max |diff| {worst:.3g}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
