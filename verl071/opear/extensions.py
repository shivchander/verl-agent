"""O-PEaR extensions for verl's trainer and actor.

Extends verl's classes via method wrapping — no installed files are modified.
Call `apply()` before importing verl.trainer.main_ppo so the extensions
take effect when Ray forks workers.

Architecture:
    - Trainer extension: injects contrastive data generation into _update_actor
    - Actor extension: adds O-PEaR loss after the standard policy update
"""
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply():
    """Apply O-PEaR extensions to verl's RayPPOTrainer and DataParallelPPOActor.

    Safe to call multiple times (no-op after the first call).
    Must be called BEFORE verl.trainer.main_ppo is imported.
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _extend_trainer()
    _extend_actor()
    logger.info("O-PEaR extensions applied")


# ---------------------------------------------------------------------------
# Trainer: inject contrastive data generation before actor update
# ---------------------------------------------------------------------------

def _extend_trainer():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    _orig_init = RayPPOTrainer.__init__
    _orig_update_actor = RayPPOTrainer._update_actor

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.opear_enabled = self.config.algorithm.get("opear", {}).get("enable", False)
        if self.opear_enabled:
            from verl071.opear.guide import OPEaRGuide
            cfg = self.config.algorithm.opear
            self.opear_guide = OPEaRGuide(
                model=cfg.get("guide_model", "gpt-5.4-nano"),
                beta=cfg.get("beta", 0.5),
            )
            self.opear_lambda = cfg.get("lambda_coef", 0.5)
            self.opear_alpha = cfg.get("alpha", 0.5)
            print(f"[O-PEaR] enabled: lambda={self.opear_lambda}, "
                  f"alpha={self.opear_alpha}, beta={cfg.get('beta', 0.5)}")

    def _update_actor(self, batch):
        if getattr(self, "opear_enabled", False):
            _generate_contrastive_data(self, batch)
        return _orig_update_actor(self, batch)

    RayPPOTrainer.__init__ = __init__
    RayPPOTrainer._update_actor = _update_actor


def _generate_contrastive_data(trainer, batch):
    """Generate contrastive pairs and attach to batch.meta_info."""
    from verl071.opear.data import reconstruct_trajectories, tokenize_contrastive_responses

    trajectories = reconstruct_trajectories(batch, trainer.tokenizer)
    if not trajectories:
        return

    selected_uids = trainer.opear_guide.select_rollouts(
        batch.non_tensor_batch.get("traj_uid", []),
        trainer.config.actor_rollout_ref.rollout.n,
    )
    selected = [t for t in trajectories if t["traj_uid"] in set(selected_uids)]
    if not selected:
        return

    pairs = trainer.opear_guide.generate_contrastive_batch(selected)
    opear_data = tokenize_contrastive_responses(
        selected, pairs, batch, trainer.tokenizer,
        max_response_length=batch.batch["responses"].shape[-1],
    )
    if opear_data is not None:
        batch.meta_info["opear_data"] = opear_data
        batch.meta_info["opear_alpha"] = trainer.opear_alpha
        batch.meta_info["opear_lambda"] = trainer.opear_lambda
        n = sum(1 for p in pairs if p is not None)
        print(f"[O-PEaR] {n}/{len(selected)} valid contrastive pairs")
    else:
        print("[O-PEaR] no valid contrastive pairs")


# ---------------------------------------------------------------------------
# Actor: add O-PEaR loss after standard policy update
# ---------------------------------------------------------------------------

def _extend_actor():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    _orig_update_policy = DataParallelPPOActor.update_policy

    def update_policy(self, data):
        opear_data = data.meta_info.get("opear_data") if hasattr(data, "meta_info") else None
        metrics = _orig_update_policy(self, data)
        if opear_data is not None:
            opear_metrics = _opear_step(self, data, opear_data)
            from verl.utils.py_functional import append_to_dict
            append_to_dict(metrics, opear_metrics)
        return metrics

    DataParallelPPOActor.update_policy = update_policy


def _opear_step(actor, data, opear_data):
    """Additional optimizer step for O-PEaR regularizer."""
    import torch
    from verl071.opear.loss import compute_opear_loss
    from verl.utils.model import compute_position_id_with_mask

    alpha = data.meta_info.get("opear_alpha", 0.5)
    lam = data.meta_info.get("opear_lambda", 0.5)
    temperature = data.meta_info.get("temperature", 1.0)
    device = next(actor.actor_module.parameters()).device

    c_ids = opear_data["compliant_input_ids"].to(device)
    c_attn = opear_data["compliant_attention_mask"].to(device)
    c_mask = opear_data["compliant_response_mask"].to(device)
    v_ids = opear_data["violating_input_ids"].to(device)
    v_attn = opear_data["violating_attention_mask"].to(device)
    v_mask = opear_data["violating_response_mask"].to(device)

    resp_len = c_mask.shape[-1]

    actor.actor_optimizer.zero_grad()

    _, c_lp = actor._forward_micro_batch(
        micro_batch={"input_ids": c_ids, "attention_mask": c_attn,
                     "position_ids": compute_position_id_with_mask(c_attn),
                     "responses": c_ids[:, -resp_len:]},
        temperature=temperature, calculate_entropy=False)

    _, v_lp = actor._forward_micro_batch(
        micro_batch={"input_ids": v_ids, "attention_mask": v_attn,
                     "position_ids": compute_position_id_with_mask(v_attn),
                     "responses": v_ids[:, -resp_len:]},
        temperature=temperature, calculate_entropy=False)

    # Align sizes
    c_lp = c_lp[:, :resp_len]
    v_lp = v_lp[:, :resp_len]
    c_rm = c_mask[:, :c_lp.shape[-1]]
    v_rm = v_mask[:, :v_lp.shape[-1]]

    loss, metrics = compute_opear_loss(c_lp, c_rm, v_lp, v_rm, alpha=alpha)
    scaled = lam * loss
    scaled.backward()

    metrics["opear/scaled_loss"] = scaled.detach().item()
    metrics["opear/lambda"] = lam
    metrics["opear/alpha"] = alpha

    grad_norm = actor._optimizer_step()
    metrics["opear/grad_norm"] = grad_norm.detach().item()

    return metrics
