# rl_env/envs/logger.py

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def _json_default(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class EnvLogger:
    def __init__(
        self,
        log_dir: str,
        enabled: bool = True,
        episode_csv_name: str = "episode_log.csv",
        step_csv_name: str = "step_log.csv",
        episode_jsonl_name: str = "episode_log.jsonl",
        save_step_log: bool = False,
        flush_every_step: bool = False,
    ) -> None:
        self.enabled = enabled
        self.log_dir = log_dir
        self.episode_csv_name = episode_csv_name
        self.step_csv_name = step_csv_name
        self.episode_jsonl_name = episode_jsonl_name
        self.save_step_log = save_step_log
        self.flush_every_step = flush_every_step

        self.episode_csv_path = os.path.join(log_dir, episode_csv_name)
        self.step_csv_path = os.path.join(log_dir, step_csv_name)
        self.episode_jsonl_path = os.path.join(log_dir, episode_jsonl_name)
        self.config_json_path = os.path.join(log_dir, "env_config.json")

        self._episode_header_written = False
        self._step_header_written = False

        self._step_file = None
        self._step_writer = None

        if self.enabled:
            ensure_dir(self.log_dir)

    def write_env_config(self, config: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with open(self.config_json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)

    def _ensure_episode_header(self) -> None:
        if self._episode_header_written or not self.enabled:
            return
        with open(self.episode_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "episode",
                    "episode_reward",
                    "episode_length",
                    "max_abs_theta",
                    "max_abs_alpha",
                    "final_theta",
                    "final_theta_dot",
                    "final_alpha",
                    "final_alpha_dot",
                    "done_reason",
                ]
            )
        self._episode_header_written = True

    def _ensure_step_writer(self) -> None:
        if not self.enabled or not self.save_step_log:
            return
        if self._step_writer is not None:
            return

        self._step_file = open(self.step_csv_path, "w", newline="", encoding="utf-8")
        self._step_writer = csv.writer(self._step_file)
        self._step_writer.writerow(
            [
                "timestamp",
                "episode",
                "step",
                "theta",
                "theta_dot",
                "alpha",
                "alpha_dot",
                "action",
                "pwm",
                "reward",
                "terminated",
                "truncated",
            ]
        )
        self._step_header_written = True
        self._step_file.flush()

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        max_abs_theta: float,
        max_abs_alpha: float,
        final_state,
        done_reason: Optional[str],
    ) -> None:
        if not self.enabled:
            return

        self._ensure_episode_header()
        timestamp = datetime.now().isoformat()
        final_theta, final_theta_dot, final_alpha, final_alpha_dot = [float(x) for x in final_state]

        with open(self.episode_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    timestamp,
                    int(episode),
                    float(episode_reward),
                    int(episode_length),
                    float(max_abs_theta),
                    float(max_abs_alpha),
                    final_theta,
                    final_theta_dot,
                    final_alpha,
                    final_alpha_dot,
                    done_reason,
                ]
            )

        with open(self.episode_jsonl_path, "a", encoding="utf-8") as f:
            rec = {
                "timestamp": timestamp,
                "episode": int(episode),
                "episode_reward": float(episode_reward),
                "episode_length": int(episode_length),
                "max_abs_theta": float(max_abs_theta),
                "max_abs_alpha": float(max_abs_alpha),
                "final_state": [final_theta, final_theta_dot, final_alpha, final_alpha_dot],
                "done_reason": done_reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_step(
        self,
        episode: int,
        step: int,
        state,
        action,
        pwm: float,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not self.enabled or not self.save_step_log:
            return

        self._ensure_step_writer()
        theta, theta_dot, alpha, alpha_dot = [float(x) for x in state]

        self._step_writer.writerow(
            [
                datetime.now().isoformat(),
                int(episode),
                int(step),
                theta,
                theta_dot,
                alpha,
                alpha_dot,
                action if not hasattr(action, "tolist") else action.tolist(),
                float(pwm),
                float(reward),
                bool(terminated),
                bool(truncated),
            ]
        )

        if self.flush_every_step and self._step_file is not None:
            self._step_file.flush()

    def close(self) -> None:
        if self._step_file is not None:
            self._step_file.flush()
            self._step_file.close()
            self._step_file = None
            self._step_writer = None