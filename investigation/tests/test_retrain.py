"""Tests for the retraining code (plan 05). Run from the repo root in the ESMDynamic environment
(it has openfold; no pytest needed, pytest also works):

    python investigation/tests/test_retrain.py [name-substring ...]

Tests needing the released V2 weights skip if the file is absent; test_cached_equals_live needs
a GPU and RETRAIN_TEST_CACHE (run by investigation/slurm/retrain_timing.sbatch).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "investigation" / "scripts"))
os.chdir(ROOT)

import retrain_lib as R  # noqa: E402

R.provide_training_imports()
V2 = Path(os.path.expanduser(R.V2_WEIGHTS))


class Skip(Exception):
    pass


def need_v2():
    if not V2.exists():
        raise Skip("V2 weights not present")


def tiny_dcm_head(n_conditions=1):
    """A real DynamicHead (real DynamicModule / Evoformer blocks), small dims."""
    from esm.esmdynamic.esmdynamic import DynamicHead
    cfg = dict(num_blocks=2, sequence_state_dim=64, pairwise_state_dim=32, sequence_head_width=16,
               pairwise_head_width=16, max_recycles=4)
    return DynamicHead("dynamic", "classification", seq_input_dim=23 + 37 * 50, seq_state_dim=64,
                       pair_input_dim=128, pair_state_dim=32, dynamic_cfg=cfg,
                       n_conditions=n_conditions, use_confidence_head=False)


def fake_structure(L=12, c_s=64, c_z=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"s_s": r(1, L, c_s), "s_z": r(1, L, L, c_z), "lddt_head": r(8, 1, L, 37, 50),
            "lm_logits": r(1, L, 23), "ptm_logits": r(1, L, L, 64), "distogram_logits": r(1, L, L, 64),
            "residue_index": torch.arange(L)[None], "mask": torch.ones(1, L, dtype=torch.long)}


# ---- the bug and the fix, on a real DynamicModule (CPU bf16 autocast reproduces it)
def _block_gradient(cache_enabled):
    torch.manual_seed(0)
    head = tiny_dcm_head()
    with torch.no_grad():   # perturb the zero-init output layers so only the cache can block gradient
        for n, p in head.named_parameters():
            if ".blocks." in n and n.endswith(R.ZERO_INIT):
                p.normal_(0, 0.02)
    model = torch.nn.Module()
    model.heads = torch.nn.ModuleDict({"dynamic": head})
    st = R.run_heads(model, fake_structure(), cache_enabled, device_type="cpu")
    st["dynamic_logits"].float().pow(2).mean().backward()
    blk = [p for n, p in head.named_parameters() if ".blocks." in n and p.dim() == 2]
    with_grad = sum(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in blk)
    assert head.prediction_linear.weight.grad is not None
    if cache_enabled:
        assert with_grad == 0, f"bug arm: {with_grad}/{len(blk)} block matrices got gradient"
    else:
        assert with_grad == len(blk), f"fixed arm: only {with_grad}/{len(blk)} block matrices got gradient"


def test_block_gradient_bug_arm():
    _block_gradient(cache_enabled=True)


def test_block_gradient_fixed_arm():
    _block_gradient(cache_enabled=False)


# ---- loss = the authors' esmdynamic_loss
def test_loss_equals_esmdynamic_loss():
    from esm.esmdynamic.training import loss as loss_mod
    g = torch.Generator().manual_seed(1)
    L, B = 10, 1
    st = {"dynamic_logits": torch.randn(B, 5, L, L, generator=g),
          "dynamic_confidence": torch.rand(B, 5, L, generator=g),
          "kinetic_logits": torch.randn(B, 5, 2, L, L, 6, generator=g),
          "kinetic_confidence": torch.rand(B, 5, L, generator=g),
          "frequency_pred": torch.rand(B, 5, L, L, generator=g),
          "frequency_residual_pred": torch.rand(B, 5, L, L, generator=g),
          "s_s": torch.zeros(1)}
    dyn = (torch.rand(B, 5, L, L, generator=g) > 0.8).float()
    kin = torch.randint(0, 6, (B, 5, 2, L, L), generator=g).float()
    freq = torch.rand(B, 5, L, L, generator=g)
    lengths = torch.tensor([L])
    kin_w = torch.rand(2, 6, generator=g) + 0.5
    total, per, outs, tgts = R.compute_loss(st, (dyn, kin, freq, lengths), R.ALL_LOSSES, 0.85, kin_w)
    ref = loss_mod.esmdynamic_loss(outs, tgts, lengths, active_heads=R.ALL_LOSSES,
                                   kin_class_weights=kin_w, alpha=0.85, gamma=2)
    assert torch.allclose(total, ref)
    assert np.isclose(np.mean(list(per.values())), float(ref), rtol=1e-6)
    p = torch.sigmoid(st["dynamic_logits"]); t = dyn
    ce = torch.nn.functional.binary_cross_entropy_with_logits(st["dynamic_logits"], t, reduction="none")
    fl = (0.85 * t + 0.15 * (1 - t)) * ce * (1 - (p * t + (1 - p) * (1 - t))) ** 2
    assert np.isclose(per["dynamic_logits"], float(fl.sum()), rtol=1e-5)


def test_masked_dynamic_loss_drops_only_near_diagonal_pairs():
    """min_sep=3: the dynamic loss is the same focal loss summed over |i-j| >= 3 only; the other
    five heads are untouched; min_sep=0 is still esmdynamic_loss exactly."""
    g = torch.Generator().manual_seed(2)
    L, B = 12, 1
    st = {"dynamic_logits": torch.randn(B, 5, L, L, generator=g),
          "dynamic_confidence": torch.rand(B, 5, L, generator=g),
          "kinetic_logits": torch.randn(B, 5, 2, L, L, 6, generator=g),
          "kinetic_confidence": torch.rand(B, 5, L, generator=g),
          "frequency_pred": torch.rand(B, 5, L, L, generator=g),
          "frequency_residual_pred": torch.rand(B, 5, L, L, generator=g),
          "s_s": torch.zeros(1)}
    dyn = (torch.rand(B, 5, L, L, generator=g) > 0.7).float()
    kin = torch.randint(0, 6, (B, 5, 2, L, L), generator=g).float()
    freq = torch.rand(B, 5, L, L, generator=g)
    lengths = torch.tensor([L - 2])                              # padding must stay masked too
    kin_w = torch.rand(2, 6, generator=g) + 0.5
    t0, p0, _, _ = R.compute_loss(st, (dyn, kin, freq, lengths), R.ALL_LOSSES, 0.85, kin_w)
    t3, p3, outs, tgts = R.compute_loss(st, (dyn, kin, freq, lengths), R.ALL_LOSSES, 0.85, kin_w, min_sep=3)
    for h in R.ALL_LOSSES:
        if h != "dynamic_logits":
            assert np.isclose(p3[h], p0[h], rtol=1e-6), h
    x, t = st["dynamic_logits"], dyn
    p = torch.sigmoid(x)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(x, t, reduction="none")
    fl = (0.85 * t + 0.15 * (1 - t)) * ce * (1 - (p * t + (1 - p) * (1 - t))) ** 2
    i = torch.arange(L)
    keep = ((i[:, None] - i[None, :]).abs() >= 3) & (i[:, None] < L - 2) & (i[None, :] < L - 2)
    assert np.isclose(p3["dynamic_logits"], float((fl * keep).sum()), rtol=1e-5)
    assert p3["dynamic_logits"] < p0["dynamic_logits"]
    assert np.isclose(float(t3), np.mean(list(p3.values())), rtol=1e-6)    # mean over heads


def test_per_protein_mean():
    """--per-protein-mean: a 256-residue window keeps its exact loss; a shorter one is scaled up by
    (elements at 256) / (its elements), per head; the pair counts match brute force."""
    for L in (5, 12, 256):
        for k in (0, 1, 3):
            i = np.arange(L)
            assert R.n_pairs(L, k) == int((np.abs(i[:, None] - i[None, :]) >= k).sum()), (L, k)

    def fake(L, seed):
        g = torch.Generator().manual_seed(seed); B = 1
        st = {"dynamic_logits": torch.randn(B, 5, L, L, generator=g),
              "dynamic_confidence": torch.rand(B, 5, L, generator=g),
              "kinetic_logits": torch.randn(B, 5, 2, L, L, 6, generator=g),
              "kinetic_confidence": torch.rand(B, 5, L, generator=g),
              "frequency_pred": torch.rand(B, 5, L, L, generator=g),
              "frequency_residual_pred": torch.rand(B, 5, L, L, generator=g), "s_s": torch.zeros(1)}
        tg = ((torch.rand(B, 5, L, L, generator=g) > 0.7).float(),
              torch.randint(0, 6, (B, 5, 2, L, L), generator=g).float(),
              torch.rand(B, 5, L, L, generator=g), torch.tensor([L]))
        return st, tg
    kin_w = torch.rand(2, 6, generator=torch.Generator().manual_seed(9)) + 0.5
    for min_sep in (0, 3):
        st, tg = fake(256, 3)
        a = R.compute_loss(st, tg, R.ALL_LOSSES, 0.85, kin_w, min_sep=min_sep)[1]
        b = R.compute_loss(st, tg, R.ALL_LOSSES, 0.85, kin_w, min_sep=min_sep, per_protein=True)[1]
        assert all(np.isclose(a[h], b[h], rtol=1e-6) for h in a), (min_sep, a, b)
        st, tg = fake(128, 4)
        a = R.compute_loss(st, tg, R.ALL_LOSSES, 0.85, kin_w, min_sep=min_sep)[1]
        b = R.compute_loss(st, tg, R.ALL_LOSSES, 0.85, kin_w, min_sep=min_sep, per_protein=True)[1]
        for h in a:
            sep = min_sep if h == "dynamic_logits" else 0
            want = (R.n_pairs(256, sep) / R.n_pairs(128, sep)) if h in R.PAIR_LOSSES else 2.0
            assert np.isclose(b[h] / a[h], want, rtol=1e-5), (h, b[h] / a[h], want)


def test_lr_schedule_and_ema():
    import train_from_v2 as T
    f = lambda s: T.lr_factor(s, warmup=10, total=110, schedule="cosine", floor=0.01)  # noqa: E731
    assert np.isclose(f(0), 0.1) and np.isclose(f(9), 1.0) and np.isclose(f(10), 1.0)
    assert np.isclose(f(60), 0.505) and np.isclose(f(110), 0.01) and np.isclose(f(500), 0.01)
    assert all(f(s) >= f(s + 1) for s in range(10, 200))
    assert T.lr_factor(1000, 0, 10, "constant", 0.01) == 1.0
    p = torch.nn.Parameter(torch.zeros(3))
    ema = T.EMA([("w", p)], decay=0.9)
    with torch.no_grad():
        p.fill_(1.0)
    ema.update()
    assert torch.allclose(ema.shadow["w"], torch.full((3,), 0.1))
    ema.swap(); assert torch.allclose(p.detach(), torch.full((3,), 0.1))
    ema.swap(); assert torch.allclose(p.detach(), torch.ones(3))


# ---- crops and sampling = DynContactDataset + WeightedRandomSampler
def test_windows_cover_every_drawable_crop():
    lengths = [100, 256, 257, 300]
    w = R.enumerate_windows(lengths)
    assert [x for x in w if x[0] == 0] == [(0, 0, 100)] and [x for x in w if x[0] == 1] == [(1, 0, 256)]
    assert [x for x in w if x[0] == 2] == [(2, 0, 256)]                 # randint(0, 1) -> start 0 only
    assert [x[1] for x in w if x[0] == 3] == list(range(44))             # randint(0, 44): 0..43
    g = torch.Generator().manual_seed(0)                                  # the dataset's own draw
    starts = {int(torch.randint(0, 300 - 256, (), generator=g)) for _ in range(5000)}
    assert starts == set(range(44))


def test_epoch_draws_depend_on_seed_only_and_follow_length():
    lengths = np.array([100, 300, 600, 50])
    a = R.draw_epoch(lengths, 4000, R.epoch_seed(0, 3, "train"))
    assert a == R.draw_epoch(lengths, 4000, R.epoch_seed(0, 3, "train"))
    assert a != R.draw_epoch(lengths, 4000, R.epoch_seed(0, 4, "train"))
    assert a != R.draw_epoch(lengths, 4000, R.epoch_seed(1, 3, "train"))
    freq = np.bincount([d for d, _ in a], minlength=4) / len(a)
    assert np.allclose(freq, lengths / lengths.sum(), atol=0.02)
    idx = {(d, s) for d, s, _ in R.enumerate_windows(lengths)}
    assert all((d, s) in idx for d, s in a)                              # every draw is cached


def test_fixed_windows_deterministic():
    lengths = [100, 400, 270]
    assert R.fixed_windows(lengths, 0) == R.fixed_windows(lengths, 0)
    for i, s, e in R.fixed_windows(lengths, 0):
        assert e - s == min(256, lengths[i]) and s < max(1, lengths[i] - 256)


def test_unpack_rcsb_row_major():
    L = 5
    m = torch.zeros(L, L, dtype=torch.uint8); m[0, 2] = m[2, 0] = 1; m[1, 4] = m[4, 1] = 1
    r, c = np.triu_indices(L, 1)
    assert torch.equal(R.unpack_rcsb(m.numpy()[r, c], L)[0], m)


def test_paper_metrics_sklearn_style():
    L = 6
    y = torch.zeros(5, L, L, dtype=torch.uint8)
    sep = (np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 3)
    y[1][torch.as_tensor(sep)] = 1                       # temperature 1: every scored pair dynamic
    prob = torch.full((1, 5, L, L), 0.9)
    st = {"dynamic_prob": prob, "frequency_pred": torch.zeros(1, 5, L, L)}
    rec = {"dyn": y, "freq": torch.zeros(5, L, L)}
    out = R.paper_metrics(st, rec)
    assert out[1]["bal_acc"] == 1.0                      # no negatives -> recall alone
    assert out[0]["bal_acc"] == 0.0 and np.isnan(out[0]["auroc"])   # no positives -> specificity


# ---- the released V2 weights: trainable sets, frozen tensors
def test_v2_at_init_set():
    need_v2()
    m = R.load_v2()
    R.check_at_init(m)
    names = R.at_init_names([n for n, _ in m.named_parameters()])
    assert len(names) == 488
    assert sum(dict(m.named_parameters())[n].numel() for n in names) == 88_737_698
    assert R.block_output_norms(m) == {"dynamic": 0.0, "kinetic": 0.0, "frequency": 0.0}


def test_set_trainable_and_frozen_stay_frozen():
    need_v2()
    m = R.load_v2()
    tr = R.set_trainable(m, "at-init")
    assert len(tr) == 488 and all(p.requires_grad == (n in tr) for n, p in m.named_parameters())
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    for p in m.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)
    opt.step()
    assert all(torch.equal(p.detach(), frozen[n]) for n, p in m.named_parameters() if n in frozen)
    assert len(R.set_trainable(m, "all")) == len(list(m.parameters()))


def test_v2_step0_is_released_model():
    """Option 1 / 2 start: our model's outputs on a structure equal the released model's
    (same weights, blocks still identity: deleting them changes nothing, as note 05)."""
    need_v2()
    m = R.load_v2().eval()
    import copy
    m2 = copy.deepcopy(m)
    for h in m2.heads.values():
        h.dynamic_module.blocks = torch.nn.ModuleList()
    st = fake_structure(L=10, c_s=1024, c_z=128, seed=3)
    with torch.no_grad():
        a = R.run_heads(m, st, True, device_type="cpu")
        b = R.run_heads(m2, st, True, device_type="cpu")
    for k in ("dynamic_logits", "kinetic_logits", "frequency_pred"):
        assert torch.equal(a[k], b[k]), k


# ---- cached features == live ESMFold (GPU)
def test_cached_equals_live():
    cache = os.environ.get("RETRAIN_TEST_CACHE")
    if not torch.cuda.is_available() or not cache:
        raise Skip("needs a GPU and RETRAIN_TEST_CACHE")
    import json
    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences
    store = R.WindowStore(cache)
    meta = json.loads((Path(cache) / "meta.json").read_text())
    fold = esm.pretrained.esmfold_v1().cuda().requires_grad_(False)
    fold.set_chunk_size(256)
    fold.train(meta["esmfold_mode"] == "train")
    model = R.load_v2(device="cuda").eval()
    model.set_chunk_size(256)
    worst, exact, n = 0.0, 0, min(20, len(store))
    for w in range(n):
        rec = store.get(w)
        seq = R.read_seq(Path(R.DATASETS["mdcath"]["folders"]) / rec["domain"])[rec["start"]:rec["end"]]
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        if meta["esmfold_mode"] == "train":
            torch.manual_seed(meta["seed"] * 1_000_003 + w)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            live = fold(aatype.cuda(), mask.cuda(), residx.cuda(), None, None)
        live = {k: live[k] for k in ("s_s", "s_z", "lm_logits", "ptm_logits", "distogram_logits", "lddt_head")}
        live.update(residue_index=residx.cuda(), mask=mask.cuda())
        cached = R.to_structure(rec, "cuda")
        exact += all(torch.equal(live[k][0], cached[k][0]) for k in ("s_s", "s_z", "ptm_logits"))
        with torch.no_grad():
            pl = R.run_heads(model, live, True)["dynamic_prob"]
            pc = R.run_heads(model, cached, True)["dynamic_prob"]
        worst = max(worst, float((pl.float() - pc.float()).abs().max()))
    print(f"cached vs live, {n} windows: features bit-identical in {exact}; "
          f"max |prob difference| {worst:.2e}")
    assert worst < 1e-3


if __name__ == "__main__":
    names = sys.argv[1:]
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and (not names or any(x in n for x in names))]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}", flush=True)
        except Skip as e:
            print(f"SKIP {n}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {n}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed or skipped")
    sys.exit(1 if failed else 0)
