"""Shared pieces of the cached two-stage retraining (plan 04): data, crop pool, cached-feature
store, the stage-1 / stage-2 heads, the forward over cached features and the loss.

Used by `cache_esmfold_features.py`, `train_faithful.py` and `investigation/tests/`.
Everything that is the authors' is imported from the package (`DynamicHead`, `ESMDynamic`,
`training/loss.py`, `training/train.py`); what is ours is here and listed in note 15.
"""
import csv
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CROP = 256
TEMPS = 5
ZERO_INIT = ("seq_attention.o_proj.weight", "mlp_seq.mlp.3.weight", "mlp_pair.mlp.3.weight",
             "sequence_to_pair.o_proj.weight", "pair_to_sequence.linear.weight",
             "tri_mul_in.linear_z.weight", "tri_mul_out.linear_z.weight",
             "tri_att_start.mha.linear_o.weight", "tri_att_end.mha.linear_o.weight")
FEATURE_KEYS = ("s_s", "s_z", "lddt_last", "lm_logits", "ptm_logits", "distogram_logits")
STAGE2_WARM = ["dynamic_logits", "kinetic_logits", "frequency_pred"]
STAGE2_FULL = STAGE2_WARM + ["dynamic_confidence", "kinetic_confidence", "frequency_residual_pred"]
DATASETS = {
    "rcsb": dict(root="investigation/data/rcsb/rcsb", folders="investigation/data/rcsb/rcsb/rcsb",
                 conditions=1),
    "mdcath": dict(root="investigation/data/mdcath", folders="investigation/data/mdcath/dataset",
                   conditions=TEMPS),
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


# ----------------------------------------------------------------------------- data
def read_ids(path):
    return [x.strip() for x in open(path) if x.strip()]


def read_seq(folder):
    """DynContactDataset._load_sequence: line 2 of a 2-line fasta."""
    with open(Path(folder) / "consensus.fasta") as f:
        return f.readlines()[1].strip()


def excluded_rcsb(report="investigation/results/15_rcsb_labels.csv"):
    """Clusters whose label count is not L(L-1)/2 (prepare_rcsb_labels.py)."""
    if not Path(report).exists():
        raise FileNotFoundError(f"{report}: run prepare_rcsb_labels.py first")
    return {r["cluster"] for r in csv.DictReader(open(report)) if r["ok"] != "1"}


def unpack_rcsb(flat, L):
    """Row-major strict upper triangle -> symmetric (1, L, L), zero diagonal."""
    flat = torch.as_tensor(flat).to(torch.uint8).flatten()
    assert flat.numel() == L * (L - 1) // 2, (flat.numel(), L)
    m = torch.zeros(L, L, dtype=torch.uint8)
    r, c = torch.triu_indices(L, L, 1)
    m[r, c] = flat
    m[c, r] = flat
    return m[None]


def load_labels(dataset, folder, L):
    """Full-length labels: dyn (C, L, L) uint8, kin (5, 2, L, L) uint8 or None,
    freq (5, L, L) float32 or None. C = 1 for RCSB (one condition), 5 for mdCATH."""
    folder = Path(folder)
    if dataset == "rcsb":
        return unpack_rcsb(torch.load(folder / "dynamic_contacts.pt"), L), None, None
    dyn = torch.load(folder / "dynamic_contacts.pt")
    kin = torch.load(folder / "kinetics.pt")
    freq = torch.load(folder / "frequency.pt")
    assert dyn.shape == (TEMPS, L, L) and kin.shape == (TEMPS, 2, L, L), (dyn.shape, kin.shape, L)
    return dyn.to(torch.uint8), kin.to(torch.uint8), freq.float()


def split_table(dataset, split):
    """ids, lengths and sampling weights for a split. RCSB: the shipped `<split>_weights.pt`
    (verified: exactly the consensus length of each cluster). mdCATH: no weight files are
    distributed, so DynContactDataset's own fallback, weight = length. Both are what
    train.py's WeightedRandomSampler draws from (normalisation does not matter)."""
    cfg = DATASETS[dataset]
    ids = read_ids(Path(cfg["root"]) / f"{split}.csv")
    lengths = np.array([len(read_seq(Path(cfg["folders"]) / i)) for i in ids])
    if dataset == "rcsb":
        w = torch.load(Path(cfg["root"]) / f"{split}_weights.pt").double().numpy()
        assert len(w) == len(ids) and np.allclose(w, lengths), "RCSB weights are not lengths"
        bad = excluded_rcsb()
        keep = np.array([i not in bad for i in ids])
        ids = [i for i, k in zip(ids, keep) if k]
        lengths, w = lengths[keep], w[keep]
    else:
        w = lengths.astype(float)
    return ids, lengths, w


def build_crop_pool(lengths, crops_per_long, seed, crop=CROP):
    """The fixed crop pool replacing a fresh crop per draw (the stated deviation). Chains
    <= crop: one crop, the whole chain. Longer: `crops_per_long` windows, each start drawn as
    DynContactDataset does, torch.randint(0, L - crop). Returns [(cluster_index, start, end)]."""
    g = torch.Generator().manual_seed(seed)
    pool = []
    for i, L in enumerate(lengths):
        L = int(L)
        if crop >= L:
            pool.append((i, 0, L))
        else:
            for _ in range(crops_per_long):
                s = int(torch.randint(0, L - crop, (), generator=g))
                pool.append((i, s, s + crop))
    return pool


def draw_plan(pool_cluster, weights, n, seed):
    """n draws as train.py makes them (cluster with probability proportional to its weight,
    with replacement), then one of that cluster's pool crops uniformly. Depends on `seed`
    only, so arms A and B see the same sequence. Returns crop ids (np.int64)."""
    pool_cluster = np.asarray(pool_cluster)
    order = np.argsort(pool_cluster, kind="stable")
    n_clusters = len(weights)
    counts = np.bincount(pool_cluster, minlength=n_clusters)
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    g = torch.Generator().manual_seed(seed)
    w = torch.as_tensor(np.asarray(weights, float))
    clusters = torch.multinomial(w, n, replacement=True, generator=g).numpy()
    u = torch.rand(n, generator=g, dtype=torch.float64).numpy()
    k = np.minimum((u * counts[clusters]).astype(np.int64), counts[clusters] - 1)
    return order[first[clusters] + k]


def plan_seed(seed, stage, epoch, split):
    return (seed * 1_000_003 + {"pretrain": 1, "finetune": 2}[stage] * 10_007
            + epoch * 2 + {"train": 0, "val": 1}[split])


# ----------------------------------------------------------------------------- cached store
class CropStore:
    """Read cached crops: <dir>/pool.csv + shard_XXXXX.pt (each a list of per-crop dicts),
    opened memory-mapped so a draw reads only that crop's tensors."""

    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        rows = list(csv.DictReader(open(self.dir / "pool.csv")))
        self.cluster_index = np.array([int(r["cluster_index"]) for r in rows])
        self.shard = np.array([int(r["shard"]) for r in rows])
        self.pos = np.array([int(r["pos"]) for r in rows])
        self.rows = rows
        self._open = {}

    def __len__(self):
        return len(self.rows)

    def get(self, crop_id):
        k = int(self.shard[crop_id])
        if k not in self._open:
            self._open[k] = torch.load(self.dir / f"shard_{k:05d}.pt", mmap=True,
                                       weights_only=True, map_location="cpu")
        rec = self._open[k][int(self.pos[crop_id])]
        assert rec["crop_id"] == crop_id, (rec["crop_id"], crop_id)
        return rec


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, store, plan):
        self.store, self.plan = store, list(plan)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, k):
        rec = self.store.get(int(self.plan[k]))
        return {key: (v.clone() if torch.is_tensor(v) else v) for key, v in rec.items()}


def loader(store, plan, workers=4):
    return torch.utils.data.DataLoader(PlanDataset(store, plan), batch_size=None, shuffle=False,
                                       num_workers=workers, pin_memory=True,
                                       persistent_workers=False, prefetch_factor=4 if workers else None)


def to_structure(rec, device):
    """A cached crop -> the `structure` dict DynamicHead.forward reads, batch of 1, float32
    (as ESMFold hands it over under autocast; the cache is fp16)."""
    f = lambda k: rec[k].to(device, non_blocking=True).float()[None]
    return {
        "s_s": f("s_s"), "s_z": f("s_z"), "lm_logits": f("lm_logits"),
        "ptm_logits": f("ptm_logits"), "distogram_logits": f("distogram_logits"),
        "lddt_head": f("lddt_last")[None],          # DynamicHead reads lddt_head[-1]
        "residue_index": rec["residue_index"].to(device)[None],
        "mask": rec["mask"].to(device)[None],
    }


def to_targets(rec, n_conditions):
    """Targets as DynContactDataset.custom_collate_fn returns them, batch of 1."""
    l = int(rec["length"])
    dyn = rec["dyn"].float()[None]
    assert dyn.shape[1] == n_conditions, (dyn.shape, n_conditions)
    kin = rec["kin"].float()[None] if "kin" in rec else torch.zeros(1, TEMPS, 2, l, l)
    freq = rec["freq"].float()[None] if "freq" in rec else torch.zeros(1, TEMPS, l, l)
    return dyn, kin, freq, torch.tensor([l])


# ----------------------------------------------------------------------------- heads
STAGE1_HEAD = dict(name="dynamic", task_type="classification", n_conditions=1, n_classes=None,
                   use_confidence_head=False, use_residual_head=False)


def seed_all(seed):
    """torch AND numpy: openfold initialises the triangle-update projections (linear_a_p /
    linear_b_p, 8 tensors per head) with scipy's truncnorm, which draws from numpy's global RNG;
    seeding torch alone gave the two arms different initial blocks."""
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_stage1(seed):
    """Stage-1 model: the dynamic head only, one condition, and V1's prediction layer
    (LayerNorm(128) + Linear(128 -> 1); V1 keys prediction_layer.0/.1). The current
    DynamicHead only has a bare Linear, so the layer is swapped in here, under the name
    DynamicHead.forward calls (`prediction_linear`). Inferred from V1's weight shapes."""
    from esm.esmdynamic.esmdynamic import ESMDynamic
    seed_all(seed)
    m = ESMDynamic(load_esmfold=False, head_definitions=[dict(STAGE1_HEAD)])
    m.heads["dynamic"].prediction_linear = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 1))
    return m


def build_stage2(seed):
    """Stage-2 model: the three released heads (5 conditions), freshly initialised."""
    from esm.esmdynamic.esmdynamic import ESMDynamic
    seed_all(seed)
    return ESMDynamic(load_esmfold=False)


HANDOVER_PREFIXES = ("seq_transition.", "pair_transition.", "dynamic_module.")


def handover(stage1_dynamic_sd, stage2_model):
    """Stage 1 -> stage 2: copy the dynamic head's transitions and DynamicModule (V2's
    dynamic head carries V1's 36 random-init block tensors bit for bit; its transitions
    correlate ~0.89 with V1's; the kinetic and frequency heads share nothing with V1, so they
    start fresh). Prediction layer and confidence head stay fresh. Returns the copied keys."""
    head = stage2_model.heads["dynamic"]
    own = head.state_dict()
    copied = []
    for k, v in stage1_dynamic_sd.items():
        if k.startswith(HANDOVER_PREFIXES):
            assert own[k].shape == v.shape, k
            own[k] = v.clone()
            copied.append(k)
    head.load_state_dict(own)
    return sorted(copied)


def v1_as_stage1_sd(v1_path):
    """V1's state dict renamed to the stage-1 head's keys (prediction_layer -> prediction_linear)."""
    sd = torch.load(v1_path, map_location="cpu")
    return {k.replace("prediction_layer.", "prediction_linear."): v for k, v in sd.items()
            if k != "dummy_buffer"}


def block_output_norms(model):
    """Per head: Frobenius norm of the zero-initialised output layers of its DCM blocks. It is
    0 at initialisation and moves only if gradient reaches the blocks."""
    out = {}
    for name, head in model.heads.items():
        tot = 0.0
        for n, p in head.named_parameters():
            if ".blocks." in n and n.endswith(ZERO_INIT):
                tot += float(p.detach().float().norm()) ** 2
        out[name] = tot ** 0.5
    return out


# ----------------------------------------------------------------------------- forward + loss
def run_heads(model, structure, cache_enabled, device_type="cuda"):
    """ESMDynamic.forward from `precomputed`, minus the native-contact step (it calls .numpy()
    on tensors that require grad; feeds no loss): every head in order, inside one bf16
    autocast region, the arm deciding `cache_enabled`."""
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, cache_enabled=cache_enabled):
        st = dict(structure)
        for head in model.heads.values():
            st = head(st, num_recycles=None)
    return st


def compute_loss(structure, targets, loss_heads, alpha, kin_w, gamma=2.0):
    """train.py's targets (build_outputs_and_targets_for_loss) and loss (esmdynamic_loss,
    the mean over active heads), with kinetic class weights cast as train.py casts them.
    Returns (total, {head: loss}, outputs, targets)."""
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
