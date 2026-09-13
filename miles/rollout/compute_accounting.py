"""Versioned logical arithmetic estimate; this is not a hardware FLOP counter."""

import json
from pathlib import Path

VERSION = "qwen35-logical-v1"


class Qwen35Flops:
    def __init__(self, config):
        self.config = config
        if config["model_type"] != "qwen3_5_text" or set(config["layer_types"]) != {
            "linear_attention",
            "full_attention",
        }:
            raise ValueError("The compute estimate requires the dense Qwen3.5 text architecture")
        h, f = config["hidden_size"], config["intermediate_size"]
        q, kv = config["num_attention_heads"] * config["head_dim"], config["num_key_value_heads"] * config["head_dim"]
        k = config["linear_num_key_heads"] * config["linear_key_head_dim"]
        v = config["linear_num_value_heads"] * config["linear_value_head_dim"]
        vh, kd, vd = config["linear_num_value_heads"], config["linear_key_head_dim"], config["linear_value_head_dim"]
        self.full_layers = config["layer_types"].count("full_attention")
        self.linear_layers = config["layer_types"].count("linear_attention")
        self.full_token = 2 * h * (2 * q + 2 * kv + q) + 6 * h * f
        self.linear_token = (
            2 * h * (2 * k + 3 * v + 2 * vh)
            + 2 * config["linear_conv_kernel_dim"] * (2 * k + v)
            + 7 * vh * kd * vd
            + 2 * vh * vd
            + 6 * h * f
        )
        self.attention_pair = 4 * q
        self.head_token = 2 * h * config["vocab_size"]

    def forward(self, end, *, cached=0, logits=None):
        if not 0 <= cached <= end:
            raise ValueError("Invalid cache token count")
        tokens = end - cached
        pairs = (end * (end + 1) - cached * (cached + 1)) // 2
        return (
            tokens * (self.full_layers * self.full_token + self.linear_layers * self.linear_token)
            + self.full_layers * self.attention_pair * pairs
            + self.head_token * (tokens if logits is None else logits)
        )

    def generation(self, sample):
        prompt = len(sample.tokens) - sample.response_length
        cached = sample.prefix_cache_info.cached_tokens
        if sample.prefix_cache_info.total_prompt_tokens != prompt or sample.response_length < 1:
            raise ValueError("Compute accounting requires one completed generation with prompt/cache counters")
        if sample.status not in (sample.Status.COMPLETED, sample.Status.TRUNCATED):
            raise ValueError("Aborted or unfinished generation has unaccounted work")
        # The last generated token has no subsequent forward pass. SGLang keeps
        # at least the last prompt token uncached to compute its output logits.
        if not 0 <= cached < prompt:
            raise ValueError("Expected SGLang cached prefix shorter than the prompt")
        return self.forward(prompt + sample.response_length - 1, cached=cached, logits=sample.response_length)


class ComputeLedger:
    def __init__(self, args):
        config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())["text_config"]
        self.estimator = Qwen35Flops(config)
        self.contract = dict(
            version=VERSION,
            architecture=config,
            budget=args.compute_budget_flops,
            checkpoints=args.compute_checkpoint_count,
            tolerance=args.compute_overshoot_tolerance,
        )
        self.output = Path(args.save)
        self.total = 0
        self.last_rollout_id = -1
        self.pending = None

    def prepare(self, rollout_id, observed, launched, accepted, parameters):
        if self.pending is not None or rollout_id != self.last_rollout_id + 1:
            raise ValueError("Compute steps must be sequential and completed exactly once")
        flat = [sample for group in observed for sample in group]
        if len(flat) != launched or len({sample.index for sample in flat}) != launched:
            raise ValueError("Every launched generation must be present exactly once in compute accounting")
        forward = sum(self.estimator.forward(len(sample.tokens)) for sample in accepted)
        stages = dict(
            generation=sum(self.estimator.generation(sample) for sample in flat),
            reference_scoring=forward,
            policy_scoring=forward,
            training_forward=forward,
            training_backward=2 * forward,
            recomputation=0,
            optimizer=15 * parameters if accepted else 0,
        )
        self.pending = dict(
            rollout_id=rollout_id,
            stages=stages,
            generated=len(flat),
            accepted=len(accepted),
            generated_tokens=sum(s.response_length for s in flat),
            trained_tokens=sum(len(s.tokens) for s in accepted),
            before=self.total,
        )

    def schedule(self):
        return dict(consumed_fraction=min(self.total / self.contract["budget"], 1.0))

    def complete(self, rollout_id):
        if self.pending is None or self.pending["rollout_id"] != rollout_id:
            raise ValueError("No matching compute step to complete")
        step = self.pending
        self.total += sum(step["stages"].values())
        self.last_rollout_id = rollout_id
        self.pending = None
        budget = self.contract["budget"]
        interval = budget / self.contract["checkpoints"]
        crossed = int(self.total // interval) > int(step["before"] // interval)
        overshoot = max(0, self.total - budget) / budget
        step.update(
            version=VERSION,
            total=self.total,
            budget=budget,
            overshoot_fraction=overshoot,
            checkpoint_due=crossed,
            budget_exhausted=self.total >= budget,
        )
        directory = self.output / "compute_steps"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{rollout_id}.json").write_text(json.dumps(step, indent=2) + "\n")
        return step

    def save(self, rollout_id):
        if self.pending is not None or rollout_id != self.last_rollout_id:
            raise ValueError("Cannot checkpoint uncompleted compute")
        path = self.output / "rollout" / f"compute_{rollout_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(contract=self.contract, total=self.total, rollout_id=rollout_id), indent=2))

    def load(self, directory, rollout_id):
        if rollout_id < 0:
            return
        state = json.loads((Path(directory) / "rollout" / f"compute_{rollout_id}.json").read_text())
        if state["contract"] != self.contract or state["rollout_id"] != rollout_id:
            raise ValueError("Compute checkpoint contract does not match this run")
        self.total, self.last_rollout_id = state["total"], rollout_id


def validate_compute_args(args):
    if getattr(args, "compute_budget_flops", None) is None:
        return
    required = dict(
        train_backend="megatron",
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        lr_decay_style="constant",
        use_dynamic_global_batch_size=True,
        use_dynamic_batch_size=True,
        use_kl_loss=True,
        compute_advantages_and_returns=True,
        optimizer="adam",
    )
    for name, value in required.items():
        if getattr(args, name) != value:
            raise ValueError(f"Compute accounting requires {name}={value}")
    disabled = [
        "use_rollout_logprobs",
        "skip_actor_forward_only",
        "use_critic",
        "fully_async",
        "partial_rollout",
        "use_fault_tolerance",
        "recompute_granularity",
        "enable_mtp_training",
        "teacher_load",
        "load_debug_rollout_data",
        "custom_reward_post_process_path",
        "dynamic_sampling_filter_path",
        "recompute_logprobs_via_prefill",
        "recompute_loss_function",
        "group_rm",
        "sglang_speculative_algorithm",
        "lr_warmup_iters",
        "lr_warmup_samples",
        "lr_warmup_fraction",
        "rollout_all_samples_process_path",
        "rollout_sample_filter_path",
    ]
    for name in disabled:
        if getattr(args, name, None):
            raise ValueError(f"Compute accounting does not support {name}")
    if args.over_sampling_batch_size != args.rollout_batch_size or not args.save or not args.ref_load:
        raise ValueError("Compute accounting requires no oversampling, reference scoring and saved state")
    if (
        args.compute_budget_flops <= 0
        or args.compute_checkpoint_count < 1
        or not 0 < args.compute_overshoot_tolerance <= 0.1
    ):
        raise ValueError("Invalid compute budget, checkpoint count or tolerance")
