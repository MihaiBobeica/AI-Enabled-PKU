"""Recovery-first staged TD3 training worker.

The first 2,000,000 environment steps are protected nominal training using the
uploaded paper-aligned TD3 setup.  Domain randomization is introduced only at
explicit stage boundaries after the nominal checkpoint has been saved.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3 import TD3

import config
from runtime import (
    TD3PaperAligned,
    aggregate_metrics,
    config_snapshot,
    dump_json,
    emit_event,
    ensure_dir,
    episode_metrics,
    extract_physical_state,
    make_raw_env,
    make_vec_env,
    reset_vec_model_state,
    set_env_randomization_level,
)


def _supported_td3_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(TD3.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


def _continuous_action_value(action) -> float:
    return float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))


def evaluate_model(
    model: TD3PaperAligned,
    env,
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    dt_policy = float(config.ENV["physical_dt"]) * int(config.ENV["action_repeat"])
    for episode in range(int(episodes)):
        obs, _ = env.reset(seed=int(config.RUN["seed"]) + 50_000 + episode)
        rewards: List[float] = []
        states: List[np.ndarray] = []
        pwms: List[float] = []
        terminated = False
        for _ in range(int(max_steps)):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, term, trunc, info = env.step(action)
            rewards.append(float(reward))
            states.append(extract_physical_state(info, next_obs))
            pwms.append(
                float(
                    info.get(
                        "effective_pwm",
                        info.get(
                            "pwm",
                            _continuous_action_value(action)
                            * float(config.ENV["pwm_limit"]),
                        ),
                    )
                )
            )
            obs = next_obs
            terminated = bool(term)
            if term or trunc:
                break
        rows.append(
            episode_metrics(
                rewards=rewards,
                states=states,
                pwms=pwms,
                terminated=terminated,
                dt_policy=dt_policy,
                capture_angle_deg=float(config.EVAL["capture_angle_deg"]),
                stable_alpha_dot_max=float(config.EVAL["stable_alpha_dot_max"]),
                stable_hold_seconds=float(config.EVAL["stable_hold_seconds"]),
            )
        )
    return aggregate_metrics(rows)


class ProgressCallback(BaseCallback):
    def __init__(self, run_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                emit_event(
                    "episode",
                    step=int(self.num_timesteps),
                    reward=float(ep["r"]),
                    length=int(ep["l"]),
                )
        return True


class PhaseEvalBestCallback(BaseCallback):
    """Keep nominal and randomized best checkpoints independently.

    Ranking order:
      1. stable success rate;
      2. original control-oriented score;
      3. mean reward.
    """

    def __init__(
        self,
        run_dir: Path,
        eval_env,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        max_eval_steps: int,
        total_timesteps: int,
        early_stop_enabled: bool,
        early_stop_start_fraction: float,
        early_stop_patience: int,
        early_stop_min_success: float,
        reward_min_delta: float,
    ):
        super().__init__(0)
        self.run_dir = run_dir
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_eval_steps = int(max_eval_steps)
        self.total_timesteps = int(total_timesteps)
        self.early_stop_enabled = bool(early_stop_enabled)
        self.early_stop_start_fraction = float(early_stop_start_fraction)
        self.early_stop_patience = int(early_stop_patience)
        self.early_stop_min_success = float(early_stop_min_success)
        self.reward_min_delta = float(reward_min_delta)
        self.next_eval = self.eval_freq
        self.phase = "nominal"
        self.phase_randomization_level = float(
            config.EVAL["nominal_eval_randomization_level"]
        )
        self.best_success = -1.0
        self.best_control_score = -np.inf
        self.best_reward = -np.inf
        self.no_improve = 0
        self.stop_reason = ""
        self.eval_rows: List[Dict[str, Any]] = []

    def _phase_dir(self) -> Path:
        return ensure_dir(
            self.run_dir
            / (
                "best_nominal_model"
                if self.phase == "nominal"
                else "best_randomized_model"
            )
        )

    def set_phase(self, phase: str, randomization_level: float) -> None:
        if phase not in {"nominal", "randomized"}:
            raise ValueError(f"Unknown evaluation phase: {phase}")
        self.phase = phase
        self.phase_randomization_level = float(randomization_level)
        self.best_success = -1.0
        self.best_control_score = -np.inf
        self.best_reward = -np.inf
        self.no_improve = 0
        set_env_randomization_level(self.eval_env, self.phase_randomization_level)
        self.eval_env.reset()
        emit_event(
            "evaluation_phase",
            phase=self.phase,
            randomization_level=self.phase_randomization_level,
            step=int(getattr(self, "num_timesteps", 0)),
        )

    def _is_improved(self, metrics: Dict[str, Any]) -> bool:
        success = float(metrics.get("stable_success_rate", 0.0))
        score = float(metrics.get("control_score", -np.inf))
        reward = float(metrics.get("mean_reward", -np.inf))
        if success > self.best_success + 1e-12:
            return True
        if abs(success - self.best_success) <= 1e-12:
            if score > self.best_control_score + 1e-12:
                return True
            if (
                abs(score - self.best_control_score) <= 1e-12
                and reward > self.best_reward + self.reward_min_delta
            ):
                return True
        return False

    def _run_eval(self) -> bool:
        metrics = evaluate_model(
            self.model,
            self.eval_env,
            self.n_eval_episodes,
            self.max_eval_steps,
        )
        metrics.update(
            step=int(self.num_timesteps),
            phase=self.phase,
            eval_randomization_level=self.phase_randomization_level,
        )
        self.eval_rows.append(metrics)
        dump_json(self.run_dir / "eval_history.json", self.eval_rows)

        improved = self._is_improved(metrics)
        success = float(metrics.get("stable_success_rate", 0.0))
        score = float(metrics.get("control_score", -np.inf))
        reward = float(metrics.get("mean_reward", -np.inf))
        if improved:
            self.best_success = success
            self.best_control_score = score
            self.best_reward = reward
            self.no_improve = 0
            best_dir = self._phase_dir()
            self.model.save(best_dir / "best_model")
            dump_json(best_dir / "best_metrics.json", metrics)
        else:
            self.no_improve += 1

        emit_event(
            "evaluation",
            step=int(self.num_timesteps),
            phase=self.phase,
            eval_randomization_level=self.phase_randomization_level,
            stable_success_rate=success,
            mean_reward=reward,
            mean_length=float(metrics.get("mean_length", 0.0)),
            reward_per_step=(
                reward / max(float(metrics.get("mean_length", 0.0)), 1.0)
            ),
            control_score=score,
            capture_rate=float(metrics.get("capture_rate", 0.0)),
            mean_abs_alpha=float(metrics.get("mean_abs_alpha", 0.0)),
            mean_abs_pwm=float(metrics.get("mean_abs_pwm", 0.0)),
            improved=improved,
            no_improve_evals=self.no_improve,
        )

        can_stop = (
            self.phase == "randomized"
            and self.early_stop_enabled
            and self.num_timesteps
            >= self.total_timesteps * self.early_stop_start_fraction
            and self.best_success >= self.early_stop_min_success
            and self.no_improve >= self.early_stop_patience
        )
        if can_stop:
            self.stop_reason = (
                f"Randomized TD3 had no effective improvement for "
                f"{self.no_improve} evaluations; best success="
                f"{self.best_success:.3f}, best control score="
                f"{self.best_control_score:.3f}."
            )
            payload = {
                "stopped": True,
                "step": int(self.num_timesteps),
                "phase": self.phase,
                "reason": self.stop_reason,
                "best_success_rate": self.best_success,
                "best_control_score": self.best_control_score,
                "best_reward": self.best_reward,
            }
            dump_json(self.run_dir / "early_stop.json", payload)
            emit_event("early_stop", **payload)
            return False
        return True

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.eval_freq
            return self._run_eval()
        return True

    def force_eval(self) -> bool:
        return self._run_eval()


class CheckpointCallback(BaseCallback):
    def __init__(self, run_dir: Path, checkpoint_freq: int):
        super().__init__(0)
        self.out_dir = ensure_dir(run_dir / "checkpoints")
        self.freq = max(1, int(checkpoint_freq))
        self.next_save = self.freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save:
            self.model.save(self.out_dir / f"td3_{self.num_timesteps:09d}")
            while self.next_save <= self.num_timesteps:
                self.next_save += self.freq
        return True


def _clear_replay(model: TD3PaperAligned) -> None:
    if model.replay_buffer is not None:
        model.replay_buffer.reset()


def _clear_optimizers(model: TD3PaperAligned) -> None:
    for network in (model.actor, model.critic):
        optimizer = getattr(network, "optimizer", None)
        if optimizer is not None:
            optimizer.state.clear()
    model._set_separate_optimizer_lrs()


def _sync_targets(model: TD3PaperAligned) -> None:
    model.actor_target.load_state_dict(model.actor.state_dict())
    model.critic_target.load_state_dict(model.critic.state_dict())


def _resolve_stage_schedule(
    total: int,
    *,
    smoke: bool = False,
) -> Tuple[List[str], List[float], List[int]]:
    if smoke:
        names = list(map(str, config.SMOKE["stage_names"]))
        levels = list(map(float, config.SMOKE["stage_levels"]))
        steps = list(map(int, config.SMOKE["stage_steps"]))
    else:
        dr = config.DOMAIN_RANDOMIZATION
        names = list(map(str, dr["training_stage_names"]))
        levels = list(map(float, dr["training_stage_levels"]))
        steps = list(map(int, dr["training_stage_steps"]))

    if not (len(names) == len(levels) == len(steps)):
        raise ValueError("Stage names, levels and steps must have equal lengths")
    if sum(steps) != int(total):
        raise ValueError(
            f"Stage steps must sum to TD3.total_timesteps={total}; got {sum(steps)}"
        )
    if any(step <= 0 for step in steps):
        raise ValueError(f"Every stage must be positive: {steps}")
    if abs(levels[0]) > 1e-12:
        raise ValueError("The first TD3 stage must have randomization level 0.0")
    if not smoke and steps[0] != int(config.DOMAIN_RANDOMIZATION["nominal_recovery_steps"]):
        raise ValueError(
            "The first stage must equal nominal_recovery_steps="
            f"{config.DOMAIN_RANDOMIZATION['nominal_recovery_steps']}"
        )
    return names, levels, steps


def _read_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _select_deployment_model(run_dir: Path) -> Tuple[Path, str, Dict[str, Any]]:
    nominal = run_dir / "best_nominal_model" / "best_model.zip"
    randomized = run_dir / "best_randomized_model" / "best_model.zip"
    nominal_metrics = _read_metrics(run_dir / "best_nominal_model" / "best_metrics.json")
    randomized_metrics = _read_metrics(
        run_dir / "best_randomized_model" / "best_metrics.json"
    )
    threshold = float(config.EVAL["selected_model_min_success_rate"])
    if (
        randomized.exists()
        and float(randomized_metrics.get("stable_success_rate", 0.0)) >= threshold
    ):
        return randomized, "randomized", randomized_metrics
    if nominal.exists():
        return nominal, "nominal_fallback", nominal_metrics
    if randomized.exists():
        return randomized, "randomized_only_available", randomized_metrics
    raise FileNotFoundError("No nominal or randomized TD3 best model was saved")


def run_training(*, smoke: bool = False) -> Dict[str, Any]:
    if os.environ.get("OVERNIGHT_APPLY_BEST") == "1":
        overrides_path = Path(__file__).resolve().parent / "best_config_overrides.json"
        if overrides_path.exists():
            data = json.loads(overrides_path.read_text(encoding="utf-8")).get("TD3", {})
            config.TD3.update(data)
            config.TD3["total_timesteps"] = int(config.TD3.get("total_timesteps", 5_000_000))
            print(f"[OVERNIGHT] applied {overrides_path.name}: {data}", flush=True)
        else:
            print("[OVERNIGHT] missing best_config_overrides.json; using config.py", flush=True)

    run_dir = Path(config.make_run_dir()).resolve()
    ensure_dir(run_dir)
    dump_json(run_dir / "config_snapshot.json", config_snapshot())

    total = int(config.SMOKE["total_timesteps"] if smoke else config.TD3["total_timesteps"])
    n_envs = 1 if smoke else int(config.TD3["n_envs"])
    if n_envs != 1:
        raise ValueError(
            "This original-aligned TD3 package intentionally uses n_envs=1; "
            "change only after a nominal baseline has converged."
        )
    stage_names, levels, lengths = _resolve_stage_schedule(total, smoke=smoke)
    seed = int(config.RUN["seed"])

    train_env = make_vec_env(
        n_envs=n_envs,
        randomization_level=levels[0],
        seed=seed,
        log_dir=run_dir / "env_logs",
    )
    expected_shape = (int(config.ENV["observation_dim"]),)
    if tuple(train_env.observation_space.shape) != expected_shape:
        raise AssertionError(
            f"Training observation_space must be {expected_shape}, got "
            f"{train_env.observation_space.shape}"
        )
    if tuple(train_env.action_space.shape) != (1,):
        raise AssertionError(
            f"TD3 action space must be continuous shape (1,), got {train_env.action_space}"
        )

    initial_eval_level = levels[0] if smoke else float(
        config.EVAL["nominal_eval_randomization_level"]
    )
    eval_env = make_raw_env(
        randomization_level=initial_eval_level,
        seed=seed + 10_000,
        max_physical_steps=int(
            config.SMOKE["max_eval_policy_steps"]
            if smoke
            else config.EVAL["max_eval_policy_steps"]
        ),
    )

    kwargs = config.build_td3_kwargs()
    if smoke:
        kwargs.update(
            buffer_size=int(config.SMOKE["buffer_size"]),
            learning_starts=int(config.SMOKE["learning_starts"]),
            batch_size=int(config.SMOKE["batch_size"]),
            train_freq=(int(config.SMOKE["train_freq"]), "step"),
            gradient_steps=int(config.SMOKE["gradient_steps"]),
            verbose=0,
            device="cpu",
        )
    action_noise = NormalActionNoise(
        mean=np.zeros(1, dtype=np.float32),
        sigma=float(config.TD3["action_noise_sigma"]) * np.ones(1, dtype=np.float32),
    )
    model = TD3PaperAligned(
        policy=TD3Policy,
        env=train_env,
        action_noise=action_noise,
        actor_learning_rate=float(config.TD3["actor_learning_rate"]),
        critic_learning_rate=float(config.TD3["critic_learning_rate"]),
        actor_grad_clip=float(config.TD3["actor_grad_clip"]),
        critic_grad_clip=float(config.TD3["critic_grad_clip"]),
        **_supported_td3_kwargs(kwargs),
    )
    replay_shape = tuple(model.replay_buffer.observations.shape)
    if replay_shape[-1] != int(config.ENV["observation_dim"]):
        raise AssertionError(
            f"Replay observation dim must be {config.ENV['observation_dim']}, "
            f"got {replay_shape}"
        )

    eval_cb = PhaseEvalBestCallback(
        run_dir,
        eval_env,
        eval_freq=int(config.SMOKE["eval_freq"] if smoke else config.EVAL["eval_freq"]),
        n_eval_episodes=int(
            config.SMOKE["n_eval_episodes"] if smoke else config.EVAL["n_eval_episodes"]
        ),
        max_eval_steps=int(
            config.SMOKE["max_eval_policy_steps"]
            if smoke
            else config.EVAL["max_eval_policy_steps"]
        ),
        total_timesteps=total,
        early_stop_enabled=False if smoke else bool(config.EVAL["early_stop_enabled"]),
        early_stop_start_fraction=float(config.EVAL["early_stop_start_fraction"]),
        early_stop_patience=int(config.EVAL["early_stop_patience_evals"]),
        early_stop_min_success=float(config.EVAL["early_stop_min_success_rate"]),
        reward_min_delta=float(config.EVAL["early_stop_reward_min_delta"]),
    )
    callbacks = CallbackList(
        [
            ProgressCallback(run_dir),
            eval_cb,
            CheckpointCallback(
                run_dir,
                int(
                    config.SMOKE["checkpoint_freq"]
                    if smoke
                    else config.EVAL["checkpoint_freq"]
                ),
            ),
        ]
    )

    emit_event(
        "stage",
        stage="training",
        status="started",
        run_dir=str(run_dir),
        algorithm="TD3",
        obs_dim=int(config.ENV["observation_dim"]),
        action_dim=1,
        replay_obs_dim=int(config.ENV["observation_dim"]),
        recovery_steps=lengths[0],
    )

    transitions: List[Dict[str, Any]] = []
    stage_models = ensure_dir(run_dir / "stage_models")
    stopped = False

    for index, (name, level, steps) in enumerate(
        zip(stage_names, levels, lengths), start=1
    ):
        if index > 1:
            if index == 2:
                recovery_dir = ensure_dir(run_dir / "recovery_model")
                model.save(recovery_dir / "nominal_2m_last")
                dump_json(
                    recovery_dir / "recovery_boundary.json",
                    {
                        "step": int(model.num_timesteps),
                        "expected_step": int(
                            lengths[0]
                            if smoke
                            else config.DOMAIN_RANDOMIZATION["nominal_recovery_steps"]
                        ),
                        "next_randomization_level": float(level),
                        "best_nominal_model": str(
                            run_dir / "best_nominal_model" / "best_model.zip"
                        ),
                    },
                )
                eval_cb.set_phase(
                    "randomized",
                    float(level if smoke else config.EVAL["eval_randomization_level"]),
                )

            set_env_randomization_level(train_env, level)
            replay_before = int(model.replay_buffer.size())
            if bool(config.DOMAIN_RANDOMIZATION["clear_replay_between_stages"]):
                _clear_replay(model)
            if bool(config.DOMAIN_RANDOMIZATION["reset_optimizer_between_stages"]):
                _clear_optimizers(model)
            if bool(config.DOMAIN_RANDOMIZATION["sync_target_between_stages"]):
                _sync_targets(model)
            if bool(config.DOMAIN_RANDOMIZATION["reset_action_noise_between_stages"]):
                try:
                    model.action_noise.reset()
                except Exception:
                    pass
            reset_vec_model_state(model)
            warmup = int(
                config.SMOKE["stage_replay_warmup_steps"]
                if smoke
                else config.DOMAIN_RANDOMIZATION["stage_replay_warmup_steps"]
            )
            model.learning_starts = int(model.num_timesteps + warmup)
            replay_after = int(model.replay_buffer.size())
            transition = {
                "stage": index,
                "stage_name": name,
                "level": float(level),
                "step": int(model.num_timesteps),
                "replay_size_before": replay_before,
                "replay_size_after": replay_after,
                "replay_cleared": bool(
                    config.DOMAIN_RANDOMIZATION["clear_replay_between_stages"]
                ),
                "optimizer_reset": bool(
                    config.DOMAIN_RANDOMIZATION["reset_optimizer_between_stages"]
                ),
                "target_synced": bool(
                    config.DOMAIN_RANDOMIZATION["sync_target_between_stages"]
                ),
                "action_noise_reset": bool(
                    config.DOMAIN_RANDOMIZATION["reset_action_noise_between_stages"]
                ),
                "warmup_steps": warmup,
            }
            transitions.append(transition)
            dump_json(run_dir / "stage_transitions.json", transitions)
            emit_event("stage_transition", **transition)

        emit_event(
            "training_stage",
            stage_index=index,
            stage_count=len(levels),
            stage_name=name,
            level=float(level),
            stage_steps=int(steps),
            stage_start_step=int(model.num_timesteps),
            replay_cleared=(
                index > 1
                and bool(config.DOMAIN_RANDOMIZATION["clear_replay_between_stages"])
            ),
        )
        model.learn(
            total_timesteps=int(steps),
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=bool(config.TD3["progress_bar"]),
            log_interval=10,
        )
        level_tag = f"{level:.2f}".replace(".", "p")
        model.save(
            stage_models
            / f"stage_{index}_{name}_level_{level_tag}_step_"
            f"{int(model.num_timesteps):09d}"
        )
        if eval_cb.stop_reason:
            stopped = True
            break

    if not eval_cb.eval_rows:
        eval_cb.model = model
        eval_cb.force_eval()

    if not (run_dir / "best_nominal_model" / "best_model.zip").exists():
        ensure_dir(run_dir / "best_nominal_model")
        model.save(run_dir / "best_nominal_model" / "best_model")
        dump_json(
            run_dir / "best_nominal_model" / "best_metrics.json",
            {"stable_success_rate": 0.0, "control_score": -1e30, "smoke": smoke},
        )

    selected_source, selected_reason, selected_metrics = _select_deployment_model(run_dir)
    generic_best = ensure_dir(run_dir / "best_model")
    shutil.copy2(selected_source, generic_best / "best_model.zip")
    dump_json(
        generic_best / "selection.json",
        {
            "selected_from": str(selected_source),
            "selection_reason": selected_reason,
            "selection_metrics": selected_metrics,
            "minimum_randomized_success_rate": float(
                config.EVAL["selected_model_min_success_rate"]
            ),
        },
    )

    model.save(run_dir / "final_model")
    shutil.copy2(generic_best / "best_model.zip", run_dir / "selected_best_model.zip")

    recovery_last = run_dir / "recovery_model" / "nominal_2m_last.zip"
    nominal_best = run_dir / "best_nominal_model" / "best_model.zip"
    randomized_best = run_dir / "best_randomized_model" / "best_model.zip"
    summary = {
        "run_dir": str(run_dir),
        "smoke": smoke,
        "algorithm": "TD3",
        "timesteps": int(model.num_timesteps),
        "observation_shape": tuple(train_env.observation_space.shape),
        "action_shape": tuple(train_env.action_space.shape),
        "replay_observation_shape": replay_shape,
        "stage_names": stage_names,
        "stage_levels": levels,
        "stage_steps": lengths,
        "stage_transitions": transitions,
        "nominal_recovery_last": str(recovery_last) if recovery_last.exists() else None,
        "best_nominal_model": str(nominal_best) if nominal_best.exists() else None,
        "best_randomized_model": (
            str(randomized_best) if randomized_best.exists() else None
        ),
        "best_model": str(generic_best / "best_model.zip"),
        "selected_best_model": str(run_dir / "selected_best_model.zip"),
        "selected_reason": selected_reason,
        "early_stopped": stopped,
        "early_stop_reason": eval_cb.stop_reason,
        "vecnormalize_created": any(run_dir.rglob("vecnormalize.pkl")),
        "input_dim": int(config.ENV["observation_dim"]),
        "continuous_action_dim": 1,
    }
    dump_json(run_dir / "training_summary.json", summary)
    (run_dir / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    emit_event("training_finished", **summary)
    train_env.close()
    eval_env.close()
    return summary
