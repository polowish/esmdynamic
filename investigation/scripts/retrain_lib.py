"""Shared pieces of the retraining (plan 05): data, the ESMFold window cache, the model built
from the released V2 heads, the trainable sets, the forward over cached features and the loss.

Used by `cache_esmfold_features.py`, `train_from_v2.py`, `retrain_timing.py` and
`investigation/tests/`. Everything that is the authors' is imported from the V2 package
(`ESMDynamic`, `DynamicHead`, `training/loss.py`, `training/train.py`'s target building);
what is ours is here and listed in plan 05 / note 15.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

CROP = 256
TEMPS = 5
V2_WEIGHTS = "~/scratch/github/esmdynamic/model_weights/hub/checkpoints/esmdynamic.pt"
ZERO_INIT = ("seq_attention.o_proj.weight", "mlp_seq.mlp.3.weight", "mlp_pair.mlp.3.weight",
             "sequence_to_pair.o_proj.weight", "pair_to_sequence.linear.weight",
             "tri_mul_in.linear_z.weight", "tri_mul_out.linear_z.weight",
             "tri_att_start.mha.linear_o.weight", "tri_att_end.mha.linear_o.weight")
FEATURE_KEYS = ("s_s", "s_z", "lddt_last", "lm_logits", "ptm_logits", "distogram_logits")
ALL_LOSSES = ["dynamic_logits", "kinetic_logits", "frequency_pred",
              "dynamic_confidence", "kinetic_confidence", "frequency_residual_pred"]
DATASETS = {
    "rcsb": dict(root="investigation/data/rcsb/rcsb", folders="investigation/data/rcsb/rcsb/rcsb"),
    "mdcath": dict(root="investigation/data/mdcath", folders="investigation/data/mdcath/dataset"),
}


# ----------------------------------------------------------------------------- environment
def provide_training_imports():
    """training/loss.py imports torchvision's sigmoid_focal_loss and train.py imports
    tensorboard; the ESMDynamic environment has neither. Same shims as check 7
    (short_training_check.py): torchvision 0.23's function line for line, a no-op writer."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from short_training_check import provide_sigmoid_focal_loss, provide_tensorboard_stub
    provide_sigmoid_focal_loss()
    provide_tensorboard_stub()


def seed_all(seed):
    """torch AND numpy: openfold initialises the triangle-update projections with scipy's
    truncnorm, which draws from numpy's global RNG."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------- data
def read_ids(path):
    return [x.strip() for x in open(path) if x.strip()]


def read_seq(folder):
    """DynContactDataset._load_sequence: line 2 of a 2-line fasta."""
    with open(Path(folder) / "consensus.fasta") as f:
        return f.readlines()[1].strip()


def excluded_rcsb(report="investigation/results/15_rcsb_labels.csv"):
    """RCSB clusters whose label count is not L(L-1)/2 (truncated in the authors' archive)."""
    return {r["cluster"] for r in csv.DictReader(open(report)) if r["ok"] != "1"}


def unpack_rcsb(flat, L):
    """RCSB labels: row-major strict upper triangle -> symmetric (1, L, L), zero diagonal."""
    flat = torch.as_tensor(flat).to(torch.uint8).flatten()
    assert flat.numel() == L * (L - 1) // 2, (flat.numel(), L)
    m = torch.zeros(L, L, dtype=torch.uint8)
    r, c = torch.triu_indices(L, L, 1)
    m[r, c] = flat
    m[c, r] = flat
    return m[None]


def load_mdcath_labels(folder, L):
    """dyn (5, L, L) uint8, kin (5, 2, L, L) uint8, freq (5, L, L) float32, as distributed."""
    folder = Path(folder)
    dyn = torch.load(folder / "dynamic_contacts.pt")
    kin = torch.load(folder / "kinetics.pt")
    freq = torch.load(folder / "frequency.pt")
    assert dyn.shape == (TEMPS, L, L) and kin.shape == (TEMPS, 2, L, L) and freq.shape == (TEMPS, L, L)
    assert torch.equal(dyn, dyn.round()) and torch.equal(kin, kin.round())
    return dyn.to(torch.uint8), kin.to(torch.uint8), freq.float()


def split_table(dataset, split):
    """ids and full lengths of a split; sampling weight = length (RCSB: the shipped
    <split>_weights.pt, verified to equal the lengths exactly; mdCATH: no weight files are
    distributed, so DynContactDataset's own fallback)."""
    cfg = DATASETS[dataset]
    ids = read_ids(Path(cfg["root"]) / f"{split}.csv")
    lengths = np.array([len(read_seq(Path(cfg["folders"]) / i)) for i in ids])
    if dataset == "rcsb":
        w = torch.load(Path(cfg["root"]) / f"{split}_weights.pt").double().numpy()
        assert len(w) == len(ids) and np.allclose(w, lengths), "RCSB weights are not lengths"
        bad = excluded_rcsb()
        keep = np.array([i not in bad for i in ids])
        ids, lengths = [i for i, k in zip(ids, keep) if k], lengths[keep]
    return ids, lengths


# ----------------------------------------------------------------------------- crops
def crop_start_bound(L, crop=CROP):
    """DynContactDataset: if crop >= L the whole chain, else start = torch.randint(0, L - crop),
    i.e. uniform on 0 .. L - crop - 1 (upper bound exclusive, so the last residue of a long chain
    is never in a crop; kept as the published code has it)."""
    return 0 if crop >= L else L - crop


def enumerate_windows(lengths, crop=CROP):
    """Every window DynContactDataset can draw: [(domain_index, start, end)]."""
    out = []
    for i, L in enumerate(lengths):
        L = int(L)
        n = crop_start_bound(L, crop)
        if n == 0:
            out.append((i, 0, L))
        else:
            out.extend((i, s, s + crop) for s in range(n))
    return out


def fixed_windows(lengths, seed, crop=CROP):
    """One window per domain, drawn once (the fixed validation set): whole chain if short,
    else a start drawn as DynContactDataset draws it."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, L in enumerate(lengths):
        L = int(L)
        n = crop_start_bound(L, crop)
        s = 0 if n == 0 else int(torch.randint(0, n, (), generator=g))
        out.append((i, s, min(L, s + crop)))
    return out


def draw_epoch(lengths, n, seed, crop=CROP):
    """One epoch of train.py's sampling: n domains with replacement, probability proportional
    to length (WeightedRandomSampler), each with a fresh crop start (DynContactDataset).
    Returns [(domain_index, start)]. A function of `seed` only."""
    g = torch.Generator().manual_seed(seed)
    w = torch.as_tensor(np.asarray(lengths, float))
    doms = torch.multinomial(w, n, replacement=True, generator=g).tolist()
    out = []
    for d in doms:
        b = crop_start_bound(int(lengths[d]), crop)
        out.append((d, 0 if b == 0 else int(torch.randint(0, b, (), generator=g))))
    return out


def epoch_seed(seed, epoch, split):
    return seed * 1_000_003 + epoch * 2 + {"train": 0, "val": 1}[split]


# ----------------------------------------------------------------------------- cache store
class WindowStore:
    """An ESMFold window cache written by cache_esmfold_features.py:
         windows.csv   window_id, domain_index, domain, start, end, length_full, shard, pos
         labels.pt     {domain: {"dyn", "kin", "freq"}} at full length
         shard_*.pt    lists of per-window feature dicts
    Files are opened memory-mapped, so a draw reads only its window."""

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.rows = list(csv.DictReader(open(self.dir / "windows.csv")))
        self.shard = np.array([int(r["shard"]) for r in self.rows])
        self.pos = np.array([int(r["pos"]) for r in self.rows])
        self.index = {(int(r["domain_index"]), int(r["start"])): k for k, r in enumerate(self.rows)}
        n = max(int(r["domain_index"]) for r in self.rows) + 1
        self.lengths = np.zeros(n, int)
        self.domains = [""] * n
        for r in self.rows:
            self.lengths[int(r["domain_index"])] = int(r["length_full"])
            self.domains[int(r["domain_index"])] = r["domain"]
        self._shards, self._labels = {}, None

    def __len__(self):
        return len(self.rows)

    def labels(self):
        if self._labels is None:
            self._labels = torch.load(self.dir / "labels.pt", mmap=True, weights_only=True)
        return self._labels

    def window_id(self, domain_index, start):
        return self.index[(int(domain_index), int(start))]

    def get(self, window_id):
        """Features (as cached) + labels cropped to the window, as DynContactDataset crops."""
        k = int(self.shard[window_id])
        if k not in self._shards:
            self._shards[k] = torch.load(self.dir / f"shard_{k:05d}.pt", mmap=True,
                                         weights_only=True, map_location="cpu")
        feats = self._shards[k][int(self.pos[window_id])]
        r = self.rows[window_id]
        assert feats["window_id"] == window_id, (feats["window_id"], window_id)
        s, e = int(r["start"]), int(r["end"])
        lab = self.labels()[r["domain"]]
        rec = {key: v.clone() for key, v in feats.items() if torch.is_tensor(v)}
        rec.update(window_id=window_id, domain=r["domain"], start=s, end=e, length=e - s,
                   dyn=lab["dyn"][:, s:e, s:e].clone(), kin=lab["kin"][:, :, s:e, s:e].clone(),
                   freq=lab["freq"][:, s:e, s:e].clone())
        return rec


class _Windows(torch.utils.data.Dataset):
    def __init__(self, store, ids):
        self.store, self.ids = store, list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        return self.store.get(self.ids[k])


def loader(store, window_ids, workers=4):
    return torch.utils.data.DataLoader(_Windows(store, window_ids), batch_size=None, shuffle=False,
                                       num_workers=workers, pin_memory=True,
                                       prefetch_factor=4 if workers else None)


def to_structure(rec, device):
    """A cached window -> the `structure` dict DynamicHead.forward reads (batch of 1), in the
    dtypes ESMFold produced them (the cache is lossless)."""
    f = lambda k: rec[k].to(device, non_blocking=True)[None]
    return {"s_s": f("s_s"), "s_z": f("s_z"), "lm_logits": f("lm_logits"),
            "ptm_logits": f("ptm_logits"), "distogram_logits": f("distogram_logits"),
            "lddt_head": f("lddt_last")[None],                # DynamicHead reads lddt_head[-1]
            "residue_index": f("residue_index"), "mask": f("mask")}


def to_targets(rec):
    """Targets as DynContactDataset.custom_collate_fn returns them (float), batch of 1."""
    return (rec["dyn"].float()[None], rec["kin"].float()[None], rec["freq"].float()[None],
            torch.tensor([int(rec["length"])]))


# ----------------------------------------------------------------------------- model
def load_v2(path=V2_WEIGHTS, device="cpu"):
    """ESMDynamic's three released heads, V2 weights loaded strictly (no ESMFold: the cache
    supplies its outputs)."""
    from esm.esmdynamic.esmdynamic import ESMDynamic
    m = ESMDynamic(load_esmfold=False)
    sd = torch.load(Path(path).expanduser(), map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "dummy_buffer"]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m.to(device)


def at_init_names(param_names):
    """Released V2 head tensors still at their initial values: every DCM block tensor (480)
    and the occupancy head's single-state path (8), which reaches its loss only through the
    blocks. Verified on the weights by `at_init_set.py` / check_at_init."""
    return sorted(n for n in param_names if ".dynamic_module.blocks." in n
                  or n.startswith("heads.frequency.seq_transition.")
                  or n.startswith("heads.frequency.dynamic_module.recycle_s_norm."))


def check_at_init(model):
    """Strict checks on the at-init set where an exact initial value exists: LayerNorm weight
    == 1 and bias == 0, zero-initialised block output layers == 0."""
    sd = dict(model.named_parameters())
    names = at_init_names(sd)
    assert len(names) == 488, len(names)
    for n in names:
        if n.endswith(ZERO_INIT):
            assert not sd[n].detach().any(), f"{n} is not zero"
    for ln in ("heads.frequency.seq_transition.0", "heads.frequency.dynamic_module.recycle_s_norm"):
        w, b = sd[f"{ln}.weight"].detach(), sd[f"{ln}.bias"].detach()
        assert torch.equal(w, torch.ones_like(w)) and not b.any(), f"{ln} is not at its LayerNorm init"


def set_trainable(model, which):
    """`at-init`: only the tensors that never trained (option 1); `all`: every head parameter
    (option 2, as train.py). Returns the trainable names; everything else requires_grad=False."""
    names = [n for n, _ in model.named_parameters()]
    keep = set(at_init_names(names)) if which == "at-init" else set(names)
    for n, p in model.named_parameters():
        p.requires_grad_(n in keep)
    return sorted(keep)


def block_output_norms(model):
    """Per head: Frobenius norm of its DCM blocks' zero-initialised output layers (0 at init;
    moves only if gradient reaches the blocks)."""
    out = {}
    for name, head in model.heads.items():
        tot = sum(float(p.detach().float().norm()) ** 2 for n, p in head.named_parameters()
                  if ".blocks." in n and n.endswith(ZERO_INIT))
        out[name] = tot ** 0.5
    return out


# ----------------------------------------------------------------------------- forward + loss
def run_heads(model, structure, cache_enabled, device_type="cuda"):
    """ESMDynamic.forward from `precomputed`, minus the native-contact step (it calls .numpy()
    on tensors that require grad; it feeds no loss): every head in order, inside one bf16
    autocast region. cache_enabled=True is train.py as released (the blocks get no gradient);
    False is the fix."""
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, cache_enabled=cache_enabled):
        st = dict(structure)
        for head in model.heads.values():
            st = head(st, num_recycles=None)
    return st


def compute_loss(structure, targets, loss_heads, alpha, kin_w, gamma=2.0):
    """train.py's targets (build_outputs_and_targets_for_loss) and loss (esmdynamic_loss: per
    head a sum over valid pairs / residues, then the mean over heads), kinetic class weights
    cast to the logits' dtype as train.py does. Returns (total, {head: loss}, outputs, targets)."""
    from esm.esmdynamic.training import loss as loss_mod
    from esm.esmdynamic.training.train import build_outputs_and_targets_for_loss
    dyn, kin, freq, lengths = targets
    device = structure["s_s"].device
    outs, tgts = build_outputs_and_targets_for_loss(structure, dyn, kin, freq, lengths,
                                                    loss_heads, device, kin_w)
    kw = None
    if kin_w is not None and "kinetic_logits" in outs:
        kw = kin_w.to(device=device, dtype=outs["kinetic_logits"].dtype)
    L = lengths.to(device)
    total = loss_mod.esmdynamic_loss(outs, tgts, L, active_heads=loss_heads,
                                     kin_class_weights=kw, alpha=alpha, gamma=gamma)
    with torch.no_grad():
        per = {h: float(loss_mod.esmdynamic_loss(outs, tgts, L, active_heads=[h],
                                                 kin_class_weights=kw, alpha=alpha, gamma=gamma))
               for h in loss_heads}
    return total, per, outs, tgts


# ----------------------------------------------------------------------------- monitoring metrics
def _auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks (as eval_protocol.auroc / sklearn)."""
    npos = int(labels.sum()); nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    r = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def paper_metrics(structure, rec, min_sep=3):
    """Per-window validation metrics under the published mdCATH protocol (notes 12-13): pairs
    |i - j| >= 3, symmetrised dynamic_prob > 0.5, scikit-learn-style balanced accuracy (a
    window with no negatives scores its recall), AUROC; occupancy RMSE on all pairs of
    frequency_pred. One value per temperature. For monitoring only: the reported numbers come
    from eval_protocol.py / eval_pairset.py."""
    prob = structure["dynamic_prob"][0].float().cpu().numpy()
    occ = structure["frequency_pred"][0].float().cpu().numpy()
    y = rec["dyn"].numpy() > 0.5
    f = rec["freq"].numpy()
    L = y.shape[-1]
    sel = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= min_sep
    out = []
    for t in range(TEMPS):
        yt, pt = y[t][sel], prob[t][sel] > 0.5
        pos, neg = yt.sum(), (~yt).sum()
        rec_ = (pt & yt).sum() / pos if pos else float("nan")
        spec = (~pt & ~yt).sum() / neg if neg else float("nan")
        ba = np.nanmean([rec_, spec]) if pos or neg else float("nan")
        out.append(dict(bal_acc=float(ba), auroc=_auroc(prob[t][sel], yt),
                        rmse=float(math.sqrt(((occ[t] - f[t]) ** 2).mean()))))
    return out
