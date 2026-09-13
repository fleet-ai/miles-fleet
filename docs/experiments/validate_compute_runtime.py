"""Validate pinned-runtime arguments, compute hooks and saved smoke tokens without starting training."""

import json
import shlex
import sys
from pathlib import Path

import torch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from miles.rollout.compute_accounting import ComputeLedger
from miles.utils.arguments import parse_args
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.types import Sample

root = Path("/sfs/neeraj/dupo-v001/smoke/neeraj-dupo-0p8b-smoke-6ae860d9")
settings = json.loads((root / "settings.json").read_text())
sys.argv = (
    ["validate"]
    + shlex.split(load_model_args("qwen3.5-0.8B"))
    + settings["argv"]
    + ["--compute-budget-flops", "2e17", "--compute-overshoot-tolerance", "0.05"]
)
args = parse_args()
print("COMPUTE_ARGUMENTS_VALIDATED")
state = ComputeLedger(args)
samples = [Sample.from_dict(s) for s in torch.load(root / "rollouts/0.pt", weights_only=False)["samples"]]
state.prepare(0, [samples], len(samples), samples, 873438784)
assert state.pending["stages"]["generation"] > 0
assert state.pending["stages"]["training_backward"] == 2 * state.pending["stages"]["training_forward"]
# Exercise the installed scheduler itself with a real optimizer; no model or GPU startup.
parameter = torch.nn.Parameter(torch.tensor(1.0))
optimizer = torch.optim.Adam([parameter], lr=2e-6)
scheduler = OptimizerParamScheduler(
    optimizer,
    init_lr=0,
    max_lr=2e-6,
    min_lr=0,
    lr_warmup_steps=0,
    lr_decay_steps=100,
    lr_decay_style="constant",
    start_wd=0.1,
    end_wd=0.1,
    wd_incr_steps=100,
    wd_incr_style="constant",
    use_checkpoint_opt_param_scheduler=True,
)
scheduler.max_lr = scheduler.min_lr = 1e-6
scheduler.step(increment=0)
assert optimizer.param_groups[0]["lr"] == 1e-6
scheduler.step(increment=8)
assert optimizer.param_groups[0]["lr"] == 1e-6
print("REAL_SCHEDULER_AND_SAVED_TOKEN_LEDGER_VALIDATED", state.pending["stages"])
