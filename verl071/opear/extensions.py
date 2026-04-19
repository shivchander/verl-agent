"""O-PEaR extensions for verl's RayPPOTrainer.

Extends the trainer via method wrapping to inject contrastive data generation
into the training loop. The actor-side O-PEaR loss is handled by a static
verl patch (see patches/apply_verl_patches.py), not by runtime patching.

Call apply() inside the Ray worker process (e.g. in OPEaRTaskRunner.run())
before RayPPOTrainer is instantiated.
"""
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply():
    """Patch RayPPOTrainer to enable O-PEaR contrastive data generation.

    Safe to call multiple times (no-op after first).
    """
    global _APPLIED
    if _APPLIED:
        return
    _APPLIED = True

    _extend_trainer()
    logger.info("O-PEaR trainer extensions applied")


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
            )
            self.opear_selection_ratio = cfg.get("selection_ratio", 0.5)
            self.opear_lambda = cfg.get("lambda_coef", 0.5)
            self.opear_beta = cfg.get("beta", 1.0)
            print(f"[O-PEaR] enabled: lambda={self.opear_lambda}, "
                  f"beta={self.opear_beta}, selection_ratio={self.opear_selection_ratio}")

    def _update_actor(self, batch):
        if getattr(self, "opear_enabled", False):
            try:
                _generate_contrastive_data(self, batch)
            except Exception as e:
                print(f"[O-PEaR] contrastive generation failed: {e}")
                import traceback
                traceback.print_exc()
        return _orig_update_actor(self, batch)

    RayPPOTrainer.__init__ = __init__
    RayPPOTrainer._update_actor = _update_actor


def _generate_contrastive_data(trainer, batch):
    """Generate contrastive pairs and attach to batch.meta_info."""
    from verl071.opear.data import select_batch_positions, reconstruct_trajectories, tokenize_contrastive_responses

    # Select first (cheap index math), then reconstruct only selected (expensive decoding)
    positions = select_batch_positions(
        batch,
        group_size=trainer.config.actor_rollout_ref.rollout.n,
        selection_ratio=trainer.opear_selection_ratio,
    )
    if not positions:
        print("[O-PEaR] no positions selected, skipping")
        return

    selected = reconstruct_trajectories(batch, trainer.tokenizer, positions)
    if not selected:
        print("[O-PEaR] no trajectories reconstructed, skipping")
        return

    import time
    t0 = time.time()
    pairs = trainer.opear_guide.generate_contrastive_batch(selected)
    guide_time = time.time() - t0

    opear_data = tokenize_contrastive_responses(
        selected, pairs, batch, trainer.tokenizer,
        max_response_length=batch.batch["responses"].shape[-1],
    )
    if opear_data is not None:
        batch.meta_info["opear_data"] = opear_data
        batch.meta_info["opear_lambda"] = trainer.opear_lambda
        batch.meta_info["opear_beta"] = trainer.opear_beta
        batch.meta_info["opear_selection_ratio"] = trainer.opear_selection_ratio
        batch.meta_info["opear_guide_time_s"] = guide_time
        # Mean assistant segments per trajectory (how multi-turn the rollouts are)
        mean_segments = sum(len(t["assistant_segments"]) for t in selected) / max(len(selected), 1)
        batch.meta_info["opear_num_segments"] = mean_segments
        n = sum(1 for p in pairs if p is not None)
        print(f"[O-PEaR] {n}/{len(selected)} valid contrastive pairs in {guide_time:.1f}s "
              f"({mean_segments:.1f} turns/traj)")
    else:
        print("[O-PEaR] no valid contrastive pairs")
