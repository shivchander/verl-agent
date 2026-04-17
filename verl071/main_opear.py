"""O-PEaR training entry point.

Subclasses verl's TaskRunner so that O-PEaR extensions are applied inside
the Ray worker process (where RayPPOTrainer actually runs), not just in
the driver.

Usage:
    python -m verl071.main_opear [hydra overrides...]
"""
import ray
import hydra

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.utils.device import auto_set_device


class OPEaRTaskRunner(TaskRunner):
    def run(self, config):
        from verl071.opear.extensions import apply
        apply()
        return super().run(config)


def _get_verl_config_path():
    import verl.trainer
    import os
    return os.path.join(os.path.dirname(verl.trainer.__file__), "config")


@hydra.main(config_path=_get_verl_config_path(), config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    task_runner_class = ray.remote(num_cpus=1)(OPEaRTaskRunner)
    run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
