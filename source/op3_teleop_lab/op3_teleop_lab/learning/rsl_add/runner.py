from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import resolve_obs_groups

from op3_teleop_lab.learning.add.config import ADDTrainingConfig

from .algorithm import RslAddPPO


class RslAddOnPolicyRunner:
    """Small RSL-RL-style runner for PPO + online ADD."""

    def __init__(
        self,
        env,
        train_cfg: dict[str, Any],
        add_cfg: ADDTrainingConfig,
        diff_dim: int,
        log_dir: str | os.PathLike[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = train_cfg
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.add_cfg = add_cfg
        self.diff_dim = diff_dim
        self.device = device
        self.env = env
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg["save_interval"])
        self.log_dir = Path(log_dir).resolve() if log_dir is not None else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], default_sets=["critic"])
        self.alg = self._construct_algorithm(obs)

        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.current_learning_iteration = 0

    def _construct_algorithm(self, obs) -> RslAddPPO:
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name", None)
        actor_critic = ActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        alg_cfg.pop("rnd_cfg", None)
        alg_cfg.pop("symmetry_cfg", None)
        alg = RslAddPPO(
            actor_critic,
            add_cfg=self.add_cfg,
            diff_dim=self.diff_dim,
            device=self.device,
            **alg_cfg,
        )
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg

    def _prepare_writer(self) -> None:
        if self.log_dir is None or self.writer is not None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            self.writer = False
            return
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._prepare_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos: list[dict[str, Any]] = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        task_rewbuffer = deque(maxlen=100)
        disc_rewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
        cur_task_reward_sum = torch.zeros_like(cur_reward_sum)
        cur_disc_reward_sum = torch.zeros_like(cur_reward_sum)
        cur_episode_length = torch.zeros_like(cur_reward_sum)

        start_iter = self.current_learning_iteration
        total_iter = start_iter + num_learning_iterations
        for it in range(start_iter, total_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if "episode" in extras:
                        ep_infos.append(extras["episode"])
                    elif "log" in extras:
                        ep_infos.append(extras["log"])

                    total_rewards = self.alg.last_total_rewards
                    disc_rewards = self.alg.last_disc_rewards
                    if total_rewards is None or disc_rewards is None:
                        raise RuntimeError("ADD rewards were not computed for the latest transition.")
                    task_rewards = self.alg.last_task_rewards
                    if task_rewards is None:
                        raise RuntimeError("Task rewards were not recorded for the latest transition.")
                    cur_reward_sum += total_rewards
                    cur_task_reward_sum += task_rewards
                    cur_disc_reward_sum += disc_rewards
                    cur_episode_length += 1

                    done_ids = torch.nonzero(dones.reshape(-1) > 0, as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[done_ids].detach().cpu().numpy().tolist())
                        task_rewbuffer.extend(cur_task_reward_sum[done_ids].detach().cpu().numpy().tolist())
                        disc_rewbuffer.extend(cur_disc_reward_sum[done_ids].detach().cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids].detach().cpu().numpy().tolist())
                        cur_reward_sum[done_ids] = 0.0
                        cur_task_reward_sum[done_ids] = 0.0
                        cur_disc_reward_sum[done_ids] = 0.0
                        cur_episode_length[done_ids] = 0.0

                collection_time = time.time() - start
                learn_start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            learn_time = time.time() - learn_start
            self.current_learning_iteration = it
            self.log(
                it=it,
                total_iter=total_iter,
                collection_time=collection_time,
                learn_time=learn_time,
                rewbuffer=rewbuffer,
                task_rewbuffer=task_rewbuffer,
                disc_rewbuffer=disc_rewbuffer,
                lenbuffer=lenbuffer,
                ep_infos=ep_infos,
                loss_dict=loss_dict,
            )
            ep_infos.clear()

            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(self.log_dir / f"model_{it}.pt")

        if self.log_dir is not None:
            self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")

    def log(
        self,
        *,
        it: int,
        total_iter: int,
        collection_time: float,
        learn_time: float,
        rewbuffer,
        task_rewbuffer,
        disc_rewbuffer,
        lenbuffer,
        ep_infos: list[dict[str, Any]],
        loss_dict: dict[str, float],
    ) -> None:
        collection_size = self.num_steps_per_env * self.env.num_envs
        self.tot_timesteps += collection_size
        self.tot_time += collection_time + learn_time
        fps = int(collection_size / max(collection_time + learn_time, 1.0e-6))
        mean_std = self.alg.policy.action_std.mean().item()

        if self.writer:
            self.writer.add_scalar("Policy/mean_noise_std", mean_std, it)
            self.writer.add_scalar("Perf/total_fps", fps, it)
            self.writer.add_scalar("Perf/collection_time", collection_time, it)
            self.writer.add_scalar("Perf/learning_time", learn_time, it)
            self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
            for key, value in loss_dict.items():
                self.writer.add_scalar(f"Loss/{key}", value, it)
            if rewbuffer:
                self.writer.add_scalar("Train/mean_add_total_reward", statistics.mean(rewbuffer), it)
                self.writer.add_scalar("Train/mean_task_reward", statistics.mean(task_rewbuffer), it)
                self.writer.add_scalar("Train/mean_disc_reward", statistics.mean(disc_rewbuffer), it)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)
            for ep_info in ep_infos:
                for key, value in ep_info.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().float().mean().item()
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(key if "/" in key else f"Episode/{key}", value, it)

        if it == 0 or (it + 1) % int(self.cfg.get("log_interval", 10)) == 0:
            mean_total = statistics.mean(rewbuffer) if rewbuffer else 0.0
            mean_task = statistics.mean(task_rewbuffer) if task_rewbuffer else 0.0
            mean_disc = statistics.mean(disc_rewbuffer) if disc_rewbuffer else 0.0
            mean_len = statistics.mean(lenbuffer) if lenbuffer else 0.0
            print(
                {
                    "iteration": it,
                    "total_iterations": total_iter,
                    "fps": fps,
                    "mean_action_std": round(mean_std, 4),
                    "mean_add_total_reward": round(mean_total, 4),
                    "mean_task_reward": round(mean_task, 4),
                    "mean_disc_reward": round(mean_disc, 4),
                    "mean_episode_length": round(mean_len, 2),
                    **{key: round(value, 6) for key, value in loss_dict.items()},
                },
                flush=True,
            )

    def save(self, path: str | os.PathLike[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                **self.alg.state_dict(),
                "iter": self.current_learning_iteration,
                "cfg": self.cfg,
                "add_cfg": self.add_cfg,
            },
            path,
        )
        print(f"Checkpoint saved: {path}", flush=True)

    def load(self, path: str | os.PathLike[str], load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.load_state_dict(checkpoint, load_optimizer=load_optimizer)
        self.current_learning_iteration = int(checkpoint.get("iter", 0))

    def train_mode(self) -> None:
        self.alg.policy.train()
        self.alg.discriminator.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        self.alg.discriminator.eval()
