"""TGPO entry point; worker setup, datasets, and lifecycle are shared with PPO."""

import hydra

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="tgpo_trainer", version_base=None)
def main(config):
    run_ppo(config)


if __name__ == "__main__":
    main()
