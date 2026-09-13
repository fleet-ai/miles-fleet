"""Optimizer dispatch for synchronous actor-only training, including empty DUPO steps."""


async def train_actor_or_skip(actor_model, rollout_id, rollout_data_pack):
    if rollout_data_pack.get("skip_training", False):
        if rollout_data_pack["sample_indices"] or rollout_data_pack["data_ref"] is not None:
            raise ValueError("An optimizer skip must have no training samples or data references")
        return False
    await actor_model.train(rollout_id, rollout_data_pack)
    return True
