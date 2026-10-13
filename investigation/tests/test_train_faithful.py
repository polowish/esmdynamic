"""CPU tests for the cached retraining (plan 04, step 2). Run from the repo root in the
ESMDynamic environment (it has openfold; no pytest needed, pytest also works):

    python investigation/tests/test_train_faithful.py [name-substring ...]

Test 3 (cached features == live ESMFold) needs a GPU and RETRAIN_TEST_CACHE: run by the timing
job (investigation/slurm/retrain_timing.sbatch); skipped otherwise.
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
V1 = Path(os.path.expanduser("~/scratch/github/esmdynamic/model_weights/esmdynamic_V1.pt"))


def tiny_dcm_head(task="classification", n_conditions=1):
    """A real DynamicHead (real DynamicModule / Evoformer blocks), small dims."""
    from esm.esmdynamic.esmdynamic import DynamicHead
    cfg = dict(num_blocks=2, sequence_state_dim=64, pairwise_state_dim=32, sequence_head_width=16,
               pairwise_head_width=16, max_recycles=4)
    return DynamicHead("dynamic", task, seq_input_dim=23 + 37 * 50, seq_state_dim=64,
                       pair_input_dim=128, pair_state_dim=32, dynamic_cfg=cfg,
                       n_conditions=n_conditions, use_confidence_head=False)


def fake_structure(L=12, c_s=64, c_z=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"s_s": r(1, L, c_s), "s_z": r(1, L, L, c_z), "lddt_head": r(8, 1, L, 37, 50),
            "lm_logits": r(1, L, 23), "ptm_logits": r(1, L, L, 64), "distogram_logits": r(1, L, L, 64),
            "residue_index": torch.arange(L)[None], "mask": torch.ones(1, L, dtype=torch.long)}


class Skip(Exception):
    pass


# ---- (1) arm A: no gradient reaches the block Linears; arm B: it does
def test_block_gradient_arm_A():
    _block_gradient(cache_enabled=True)


def test_block_gradient_arm_B():
    _block_gradient(cache_enabled=False)


def _block_gradient(cache_enabled):
    torch.manual_seed(0)
    head = tiny_dcm_head()
    # zero-initialised output layers block gradient to everything upstream at step 0 by design;
    # perturb them so that "no gradient" can only come from the autocast cache
    with torch.no_grad():
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
    assert head.seq_transition[1].weight.grad is not None
    if cache_enabled:
        assert with_grad == 0, f"arm A: {with_grad}/{len(blk)} block matrices got gradient"
    else:
        assert with_grad == len(blk), f"arm B: only {with_grad}/{len(blk)} block matrices got gradient"


# ---- (2) our loss on a fake batch == esmdynamic_loss, and = mean of the per-head terms
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
    total, per, outs, tgts = R.compute_loss(st, (dyn, kin, freq, lengths), R.STAGE2_FULL, 0.85, kin_w)
    ref = loss_mod.esmdynamic_loss(outs, tgts, lengths, active_heads=R.STAGE2_FULL,
                                   kin_class_weights=kin_w, alpha=0.85, gamma=2)
    assert torch.allclose(total, ref)
    assert np.isclose(np.mean(list(per.values())), float(ref), rtol=1e-6)
    # the focal term itself, by hand (torchvision's definition), alpha 0.85
    p = torch.sigmoid(st["dynamic_logits"]); t = dyn
    ce = torch.nn.functional.binary_cross_entropy_with_logits(st["dynamic_logits"], t, reduction="none")
    fl = (0.85 * t + 0.15 * (1 - t)) * ce * (1 - (p * t + (1 - p) * (1 - t))) ** 2
    assert np.isclose(per["dynamic_logits"], float(fl.sum()), rtol=1e-5)


# ---- (4) identical draw sequence for A and B at the same seed; different across seeds/epochs
def test_plan_depends_on_seed_only():
    lengths = np.array([100, 300, 600, 256, 257, 50])
    pool = R.build_crop_pool(lengths, 2, seed=0)
    assert pool == R.build_crop_pool(lengths, 2, seed=0)
    assert [p for p in pool if p[0] == 0] == [(0, 0, 100)]          # short chain: one whole crop
    for i, s, e in pool:
        assert e - s == min(256, lengths[i]) and 0 <= s and (lengths[i] <= 256 or s < lengths[i] - 256)
    cl = [p[0] for p in pool]
    a = R.draw_plan(cl, lengths, 5000, R.plan_seed(0, "pretrain", 3, "train"))
    b = R.draw_plan(cl, lengths, 5000, R.plan_seed(0, "pretrain", 3, "train"))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, R.draw_plan(cl, lengths, 5000, R.plan_seed(0, "pretrain", 4, "train")))
    assert not np.array_equal(a, R.draw_plan(cl, lengths, 5000, R.plan_seed(1, "pretrain", 3, "train")))
    # clusters drawn in proportion to length
    freq = np.bincount(np.array(cl)[a], minlength=len(lengths)) / len(a)
    assert np.allclose(freq, lengths / lengths.sum(), atol=0.02)


def test_unpack_rcsb_row_major():
    L = 5
    m = torch.zeros(L, L, dtype=torch.uint8); m[0, 2] = m[2, 0] = 1; m[1, 4] = m[4, 1] = 1
    r, c = np.triu_indices(L, 1)
    assert torch.equal(R.unpack_rcsb(m.numpy()[r, c], L)[0], m)


# ---- (5) stage-1 -> stage-2 hand-over copies exactly the intended tensors
def test_handover_exact():
    s1 = R.build_stage1(seed=0)
    s2 = R.build_stage2(seed=1)
    before = {k: v.clone() for k, v in s2.state_dict().items()}
    sd1 = s1.heads["dynamic"].state_dict()
    copied = R.handover(sd1, s2)
    after = s2.state_dict()
    expected = sorted(k for k in sd1 if k.startswith(R.HANDOVER_PREFIXES))
    assert copied == expected and len(copied) == 178            # 12 transition + 6 DCM + 160 block
    for k in after:
        if k.startswith("heads.dynamic.") and k[len("heads.dynamic."):] in copied:
            assert torch.equal(after[k], sd1[k[len("heads.dynamic."):]]), k
        else:
            assert torch.equal(after[k], before[k]), f"{k} changed"
    assert s1.heads["dynamic"].prediction_linear[1].weight.shape == (1, 128)
    assert s2.heads["dynamic"].prediction_linear.weight.shape == (5, 128)


def test_same_init_for_both_arms():
    for build in (R.build_stage1, R.build_stage2):
        a = build(seed=0).state_dict()
        np.random.rand(7); torch.rand(3)                 # disturb both global RNGs in between
        b = build(seed=0).state_dict()
        diff = [k for k in a if not torch.equal(a[k], b[k])]
        assert not diff, diff[:5]


def test_v1_loads_into_stage1_head():
    """V1's 182 tensors load into the stage-1 head with strict=True: the head we pre-train is
    V1's architecture exactly (1 condition, LayerNorm + Linear prediction layer, no confidence)."""
    if not V1.exists():
        raise Skip("V1 weights not present")
    s1 = R.build_stage1(seed=0)
    s1.heads["dynamic"].load_state_dict(R.v1_as_stage1_sd(V1), strict=True)


# ---- (3) cached features give the same head outputs as live ESMFold (GPU)
def test_cached_equals_live():
    cache = os.environ.get("RETRAIN_TEST_CACHE")
    if not torch.cuda.is_available() or not cache:
        raise Skip("needs a GPU and RETRAIN_TEST_CACHE")
    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences
    store = R.CropStore(cache)
    meta = __import__("json").loads((Path(cache) / "meta.json").read_text())
    fold = esm.pretrained.esmfold_v1().cuda().requires_grad_(False)
    fold.set_chunk_size(256)
    fold.train(meta["esmfold_mode"] == "train")
    model = R.build_stage2(0)
    model.load_state_dict(torch.load(os.path.expanduser(
        "~/scratch/github/esmdynamic/model_weights/hub/checkpoints/esmdynamic.pt"), map_location="cpu"),
        strict=False)
    model.cuda().set_chunk_size(256)
    model.eval()
    worst = 0.0
    for c in range(min(20, len(store))):
        rec = store.get(c)
        seq = R.read_seq(Path(R.DATASETS[meta["dataset"]]["folders"]) / rec["cluster"])[rec["start"]:rec["end"]]
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        if meta["esmfold_mode"] == "train":
            torch.manual_seed(meta["seed"] * 1_000_003 + c)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            live = fold(aatype.cuda(), mask.cuda(), residx.cuda(), None, None)
            live["mask"] = mask.cuda()
            out_live = R.run_heads(model, live, True)["dynamic_logits"].float()
            out_cache = R.run_heads(model, R.to_structure(rec, "cuda"), True)["dynamic_logits"].float()
        d = float((torch.sigmoid(out_live) - torch.sigmoid(out_cache)).abs().max())
        worst = max(worst, d)
    print("max |prob(live) - prob(cached)| over 20 crops:", worst)
    assert worst < 2e-2


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
