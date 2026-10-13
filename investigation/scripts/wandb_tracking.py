"""Weights & Biases tracking for train_from_v2.py.

Compute nodes have no internet, so runs log OFFLINE (default) into
~/scratch/esmdyn_retrain/wandb; sync from the login node:

    WANDB_API_KEY=$(cat ~/scratch/tool_caches/wandb/api_key) \
      PYTHONPATH=~/scratch/pylibs/wandb python -m wandb sync --entity sophia-madejski \
      ~/scratch/esmdyn_retrain/wandb/wandb/offline-run-*

wandb itself lives in ~/scratch/pylibs/wandb (pip --target; not in the shared environment);
every wandb directory is on scratch (nothing in $HOME).

Logged: the run config; per optimiser step the loss per head and the block output-layer norm
per head; per epoch train/val losses, validation balanced accuracy, and per head and module
group (transitions, DCM outside the blocks, blocks, prediction layer, confidence/residual) the
parameter norm and the distance from initialisation, plus histograms of the blocks' zero-init
output layers; checkpoints as model artifacts (the current best and last every
`artifact_every` epochs when they changed, and the final best / last / warm_last).
"""
import os
import subprocess
from pathlib import Path

import torch

SCRATCH = Path(os.path.expanduser("~/scratch"))
ENV_DEFAULTS = {
    "WANDB_DIR": SCRATCH / "esmdyn_retrain" / "wandb",
    "WANDB_CACHE_DIR": SCRATCH / "tool_caches" / "wandb" / "cache",
    "WANDB_CONFIG_DIR": SCRATCH / "tool_caches" / "wandb" / "config",
    "WANDB_DATA_DIR": SCRATCH / "tool_caches" / "wandb" / "data",
    "WANDB_ARTIFACT_DIR": SCRATCH / "tool_caches" / "wandb" / "artifacts",
}


def group_of(name):
    if name.startswith(("seq_transition.", "pair_transition.")):
        return "transitions"
    if ".blocks." in name:
        return "blocks"
    if name.startswith("dynamic_module."):
        return "dcm_other"
    if name.startswith("prediction_linear"):
        return "prediction"
    return "aux"                     # confidence_head / residual_head


class Tracker:
    def __init__(self, mode, project, out, run_name, group, tags, config, model, artifact_every=10):
        self.on = mode != "off"
        self.out, self.every = Path(out), artifact_every
        self.logged = {}
        if not self.on:
            return
        for k, v in ENV_DEFAULTS.items():
            os.environ.setdefault(k, str(v))
            Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_MODE"] = mode
        import wandb
        self.wandb = wandb
        id_file = self.out / "wandb_id.txt"
        import secrets   # wandb 0.30 has no wandb.util.generate_id; ids are [a-z0-9]
        run_id = id_file.read_text().strip() if id_file.exists() else secrets.token_hex(4)
        id_file.write_text(run_id)
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:  # noqa: BLE001
            sha = None
        # entity: the user's PERSONAL account (their key's default is the UCSF org entity);
        # override with WANDB_ENTITY
        self.run = wandb.init(project=project, entity=os.environ.get("WANDB_ENTITY", "sophia-madejski"),
                              id=run_id, resume="allow", name=run_name, group=group,
                              tags=tags, config=dict(config, git_sha=sha), dir=os.environ["WANDB_DIR"])
        self.run.define_metric("epoch/*", step_metric="epoch/epoch")
        init_path = self.out / "init_heads.pt"     # for distance-from-initialisation, survives resume
        if not init_path.exists():
            torch.save(model.heads.state_dict(), init_path)
        self.init = torch.load(init_path, map_location="cpu")

    def step(self, rec, global_step):
        if self.on:
            self.run.log({f"step/{k}": v for k, v in rec.items() if isinstance(v, (int, float))},
                         step=global_step)

    def epoch(self, hist, model, global_step):
        if not self.on:
            return
        log = {f"epoch/{k}": v for k, v in hist.items() if isinstance(v, (int, float))}
        log["epoch/epoch"] = hist["epoch"]
        sq_norm, sq_delta = {}, {}
        for k, v in model.heads.state_dict().items():
            head, name = k.split(".", 1)
            key = f"{head}/{group_of(name)}"
            v = v.detach().float().cpu()
            sq_norm[key] = sq_norm.get(key, 0.0) + float(v.norm()) ** 2
            sq_delta[key] = sq_delta.get(key, 0.0) + float((v - self.init[k].float()).norm()) ** 2
        for key in sq_norm:
            log[f"weights/norm/{key}"] = sq_norm[key] ** 0.5
            log[f"weights/delta_from_init/{key}"] = sq_delta[key] ** 0.5
        from retrain_lib import ZERO_INIT
        for head_name, head in model.heads.items():
            w = [p.detach().float().flatten().cpu() for n, p in head.named_parameters()
                 if ".blocks." in n and n.endswith(ZERO_INIT)]
            log[f"weights/hist/{head_name}/block_output_layers"] = self.wandb.Histogram(torch.cat(w).numpy())
        self.run.log(log, step=global_step)
        if self.every and hist["epoch"] % self.every == 0:
            self.checkpoints(final=False, epoch=hist["epoch"])

    def checkpoints(self, final, epoch=None):
        """Log best.pt / last heads / warm_last.pt as artifacts when their content changed."""
        if not self.on:
            return
        cfg = self.run.config
        name = f"heads-{cfg['stage']}-{cfg['arm']}-s{cfg['seed']}"
        for fname, alias in (("best.pt", "best"), ("last.pt", "latest"), ("warm_last.pt", "warm-last")):
            p = self.out / fname
            if not p.exists():
                continue
            stamp = p.stat().st_mtime_ns
            if self.logged.get(fname) == stamp:
                continue
            path = p
            if fname == "last.pt":           # heads only, without optimiser state
                path = self.out / "last_heads.pt"
                torch.save({"heads": torch.load(p, map_location="cpu", weights_only=False)["heads"]}, path)
            art = self.wandb.Artifact(name, type="model",
                                      metadata=dict(file=fname, epoch=epoch, final=final))
            art.add_file(str(path), name=fname)
            aliases = [alias] + ([f"epoch-{epoch}"] if epoch else []) + (["final"] if final else [])
            self.run.log_artifact(art, aliases=aliases)
            self.logged[fname] = stamp

    def finish(self, summary=None, final=True):
        if self.on:
            if summary:
                self.run.summary.update(summary)
            self.checkpoints(final=final)
            self.run.finish()
