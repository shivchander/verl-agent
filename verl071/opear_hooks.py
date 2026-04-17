"""O-PEaR monkey-patches for the installed verl package.

Call `apply_patches()` BEFORE ray.init() so patches propagate to workers
via fork. All O-PEaR logic lives here — no modifications to installed verl.
"""
import logging

logger = logging.getLogger(__name__)

_PATCHES_APPLIED = False


def apply_patches():
    """Apply O-PEaR patches to verl's RayPPOTrainer and DataParallelPPOActor."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True

    _patch_trainer()
    _patch_actor()
    logger.info("O-PEaR patches applied to verl")


# ---------------------------------------------------------------------------
# Trainer patch: inject O-PEaR phase into the training loop
# ---------------------------------------------------------------------------

def _patch_trainer():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    _original_init = RayPPOTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)

        # Initialize O-PEaR if enabled in config
        self.opear_enabled = self.config.algorithm.get("opear", {}).get("enable", False)
        if self.opear_enabled:
            from verl071.opear.guide import OPEaRGuide
            opear_cfg = self.config.algorithm.opear
            self.opear_guide = OPEaRGuide(
                model=opear_cfg.get("guide_model", "gpt-5.4-nano"),
                beta=opear_cfg.get("beta", 0.5),
            )
            self.opear_lambda = opear_cfg.get("lambda_coef", 0.5)
            self.opear_alpha = opear_cfg.get("alpha", 0.5)
            print(f"O-PEaR enabled: lambda={self.opear_lambda}, alpha={self.opear_alpha}, "
                  f"beta={opear_cfg.get('beta', 0.5)}")

    RayPPOTrainer.__init__ = _patched_init

    # Patch _update_actor to inject O-PEaR contrastive generation before the call
    _original_update_actor = RayPPOTrainer._update_actor

    def _patched_update_actor(self, batch):
        if getattr(self, "opear_enabled", False):
            _inject_opear_data(self, batch)
        return _original_update_actor(self, batch)

    RayPPOTrainer._update_actor = _patched_update_actor


def _inject_opear_data(trainer, batch):
    """Generate O-PEaR contrastive pairs and pack them into batch.meta_info."""
    from verl071.opear.data import reconstruct_trajectories, tokenize_contrastive_responses

    trajectories = reconstruct_trajectories(batch, trainer.tokenizer)
    if not trajectories:
        return

    selected_uids = trainer.opear_guide.select_rollouts(
        batch.non_tensor_batch.get("traj_uid", []),
        trainer.config.actor_rollout_ref.rollout.n,
    )
    selected_trajs = [t for t in trajectories if t["traj_uid"] in set(selected_uids)]
    if not selected_trajs:
        return

    pairs = trainer.opear_guide.generate_contrastive_batch(selected_trajs)
    opear_data = tokenize_contrastive_responses(
        selected_trajs, pairs, batch, trainer.tokenizer,
        max_response_length=batch.batch["responses"].shape[-1],
    )

    if opear_data is not None:
        batch.meta_info["opear_data"] = opear_data
        batch.meta_info["opear_alpha"] = trainer.opear_alpha
        batch.meta_info["opear_lambda"] = trainer.opear_lambda
        num_valid = sum(1 for p in pairs if p is not None)
        print(f"[O-PEaR] {num_valid}/{len(selected_trajs)} valid contrastive pairs")
    else:
        print("[O-PEaR] No valid contrastive pairs generated")


# ---------------------------------------------------------------------------
# Actor patch: add O-PEaR loss after DrGRPO policy loss
# ---------------------------------------------------------------------------

def _patch_actor():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    _original_update_policy = DataParallelPPOActor.update_policy

    def _patched_update_policy(self, data):
        # Save reference before the original method reassigns 'data' internally
        opear_data = data.meta_info.get("opear_data") if hasattr(data, "meta_info") else None

        # Run the original DrGRPO update
        metrics = _original_update_policy(self, data)

        # Add O-PEaR loss if contrastive data is available
        if opear_data is not None:
            opear_metrics = _compute_opear_step(self, data, opear_data)
            if opear_metrics:
                from verl.utils.py_functional import append_to_dict
                append_to_dict(metrics, opear_metrics)

        return metrics

    DataParallelPPOActor.update_policy = _patched_update_policy


def _compute_opear_step(actor, data, opear_data):
    """Forward pass on contrastive data and compute O-PEaR loss."""
    import torch
    from verl071.opear.loss import compute_opear_loss
    from verl.utils.model import compute_position_id_with_mask

    opear_alpha = data.meta_info.get("opear_alpha", 0.5)
    opear_lambda = data.meta_info.get("opear_lambda", 0.5)
    temperature = data.meta_info.get("temperature", 1.0)

    device = next(actor.actor_module.parameters()).device

    c_input_ids = opear_data["compliant_input_ids"].to(device)
    c_attn_mask = opear_data["compliant_attention_mask"].to(device)
    c_resp_mask = opear_data["compliant_response_mask"].to(device)
    v_input_ids = opear_data["violating_input_ids"].to(device)
    v_attn_mask = opear_data["violating_attention_mask"].to(device)
    v_resp_mask = opear_data["violating_response_mask"].to(device)

    c_position_ids = compute_position_id_with_mask(c_attn_mask)
    v_position_ids = compute_position_id_with_mask(v_attn_mask)

    resp_len = c_resp_mask.shape[-1]
    c_responses = c_input_ids[:, -resp_len:]
    v_responses = v_input_ids[:, -resp_len:]

    actor.actor_optimizer.zero_grad()

    # Forward passes
    c_micro = {"input_ids": c_input_ids, "attention_mask": c_attn_mask,
               "position_ids": c_position_ids, "responses": c_responses}
    _, c_log_probs = actor._forward_micro_batch(
        micro_batch=c_micro, temperature=temperature, calculate_entropy=False)

    v_micro = {"input_ids": v_input_ids, "attention_mask": v_attn_mask,
               "position_ids": v_position_ids, "responses": v_responses}
    _, v_log_probs = actor._forward_micro_batch(
        micro_batch=v_micro, temperature=temperature, calculate_entropy=False)

    # Align sizes
    c_lp = c_log_probs[:, :c_resp_mask.shape[-1]]
    v_lp = v_log_probs[:, :v_resp_mask.shape[-1]]
    c_rm = c_resp_mask[:, :c_lp.shape[-1]]
    v_rm = v_resp_mask[:, :v_lp.shape[-1]]

    opear_loss, opear_metrics = compute_opear_loss(
        compliant_log_probs=c_lp, compliant_mask=c_rm,
        violating_log_probs=v_lp, violating_mask=v_rm,
        alpha=opear_alpha)

    scaled_loss = opear_lambda * opear_loss
    scaled_loss.backward()

    opear_metrics["opear/scaled_loss"] = scaled_loss.detach().item()
    opear_metrics["opear/lambda"] = opear_lambda
    opear_metrics["opear/alpha"] = opear_alpha

    grad_norm = actor._optimizer_step()
    opear_metrics["opear/grad_norm"] = grad_norm.detach().item()

    return opear_metrics
