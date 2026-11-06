"""Thin integration with verl's sole training loop."""

from functools import partial
from importlib import import_module

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.tgpo.refinement import RefinementConfig, refine_rollouts
from verl.trainer.tgpo.batch_ops import attach_reward_info


def _callable(path):
    module, name = path.rsplit(".", 1)
    return getattr(import_module(module), name)


class TGPOTrainer(RayPPOTrainer):
    """Refine rollouts before shared probability recomputation and GRPO update."""

    def postprocess_rollouts(self, batch, reward_tensor, reward_info, metrics):
        cfg = self.config.tgpo
        task = self.config.task
        # Canonical parquet stores context once in extra_info. Promote only the
        # configured fields at the trainer boundary for grouping and task repair.
        for field in task.get("metadata_fields", []):
            if field in batch.non_tensor_batch:
                continue
            extras = batch.non_tensor_batch.get("extra_info", [])
            if len(extras) != len(batch) or any(field not in record for record in extras):
                raise ValueError(f"TGPO requires {field!r} in every extra_info record")
            attach_reward_info(batch, {field: [record[field] for record in extras]})
        kwargs = OmegaConf.to_container(task.repair.get("kwargs", {}), resolve=True)
        repair = partial(_callable(task.repair.callable), **kwargs)
        formatter = partial(_callable(task.prefix_formatter), output_format=task.output_format)
        generate = (self.async_rollout_manager.generate_sequences if self.async_rollout_mode
                    else self.actor_rollout_wg.generate_sequences)
        final, reward_tensor, reward_info, tgpo_metrics = refine_rollouts(
            batch, reward_tensor, reward_info,
            tokenizer=self.tokenizer, generate=generate,
            reward=partial(compute_reward, reward_fn=self.reward_fn), repair=repair, format_prefix=formatter,
            config=RefinementConfig(
                rollout_n=self.config.actor_rollout_ref.rollout.n,
                select_num=cfg.select_num,
                continuations_per_sample=cfg.continuations_per_sample,
                reward_threshold=cfg.reward_threshold,
                position_policy=cfg.position_policy,
            ),
            generation_divisor=self.actor_rollout_wg.world_size,
        )
        metrics.update(tgpo_metrics)
        return final, reward_tensor, reward_info
