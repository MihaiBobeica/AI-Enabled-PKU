"""Custom DQN trainer that reproduces RN_DQN(3).m before adding DR.

This deliberately does not call SB3's DQN update path. The MATLAB project uses
three distinct networks (online, target and a 50-step behavior snapshot), a
specific 0.01-normal initialization and vanilla target-network max backup.
Implementing that loop directly is the only reliable way to preserve those
semantics while keeping the GUI/package workflow.
"""
from __future__ import annotations

import csv
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import config
from runtime import (
    MatlabAlignedRIPEnv,
    ReplayBuffer,
    build_network,
    checkpoint_payload,
    choose_device,
    config_snapshot,
    copy_model,
    dqn_update,
    dump_json,
    emit_event,
    ensure_dir,
    episode_summary,
    epsilon_at,
    evaluate_network,
    export_c_header,
    greedy_action,
    make_optimizer,
    save_checkpoint,
    seed_everything,
)


def _resolve_schedule(smoke: bool) -> Tuple[List[str], List[float], List[int], int]:
    if smoke:
        levels = list(map(float, config.SMOKE["stage_levels"]))
        steps = list(map(int, config.SMOKE["stage_steps"]))
        names = ["smoke_nominal" if level == 0 else f"smoke_dr_{level:.2f}" for level in levels]
        total = int(config.SMOKE["total_timesteps"])
    else:
        names = list(map(str, config.DOMAIN_RANDOMIZATION["training_stage_names"]))
        levels = list(map(float, config.DOMAIN_RANDOMIZATION["training_stage_levels"]))
        steps = list(map(int, config.DOMAIN_RANDOMIZATION["training_stage_steps"]))
        total = int(config.DQN["total_timesteps"])
    if not (len(names) == len(levels) == len(steps)):
        raise ValueError("training_stage_names/levels/steps must have equal lengths")
    if sum(steps) != total:
        raise ValueError(f"stage steps sum to {sum(steps):,}, expected total_timesteps={total:,}")
    if not levels or levels[0] != 0.0:
        raise ValueError("The first nominal stage must have level 0.0")
    return names, levels, steps, total


def _stage_epsilon(global_step: int, stage_step: int, stage_index: int) -> float:
    if stage_index == 0 or not bool(config.DQN["reset_exploration_each_randomized_stage"]):
        return epsilon_at(global_step)
    start = float(config.DQN["randomized_stage_initial_eps"])
    final = float(config.DQN["randomized_stage_final_eps"])
    decay = float(config.DQN["randomized_stage_exploration_decay"])
    return final + (start - final) * math.exp(-float(stage_step) / max(decay, 1e-12))


def _write_csv_header(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(columns)


def _append_csv(path: Path, values: Sequence[Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(values)


def _is_better(metrics: Dict[str, Any], best: Dict[str, Any] | None) -> bool:
    if best is None:
        return True
    success = float(metrics["stable_success_rate"])
    best_success = float(best["stable_success_rate"])
    if success > best_success + 1e-12:
        return True
    if abs(success - best_success) <= 1e-12:
        delta = float(config.EVAL["early_stop_reward_min_delta"])
        return float(metrics["mean_reward_per_step"]) > float(best["mean_reward_per_step"]) + delta
    return False


def _make_payload(
    online, target, behavior, optimizer, replay, *, global_step: int, episode: int,
    stage_index: int, stage_name: str, stage_level: float, stats: Dict[str, Any]
):
    return checkpoint_payload(
        online, target, behavior, optimizer,
        global_step=global_step,
        episode=episode,
        stage_index=stage_index,
        stage_name=stage_name,
        stage_level=stage_level,
        stats=stats,
        replay=replay,
    )


def run_training(smoke: bool = False) -> Dict[str, Any]:
    seed = int(config.RUN["seed"])
    seed_everything(seed)
    torch.set_num_threads(max(1, int(config.RUN.get("torch_num_threads", 4))))
    if bool(config.RUN.get("torch_deterministic", False)):
        torch.use_deterministic_algorithms(True)

    names, levels, stage_steps, total_steps = _resolve_schedule(smoke)
    run_dir = ensure_dir(Path(config.make_run_dir()))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    recovery_dir = ensure_dir(run_dir / "recovery_model")
    nominal_best_dir = ensure_dir(run_dir / "best_nominal_model")
    randomized_best_dir = ensure_dir(run_dir / "best_randomized_model")
    deploy_dir = ensure_dir(run_dir / "deploy")

    dump_json(run_dir / "config_snapshot.json", config_snapshot())
    dump_json(run_dir / "stage_schedule.json", {
        "names": names, "levels": levels, "steps": stage_steps, "total": total_steps,
    })

    device = choose_device()
    rng = np.random.default_rng(seed)
    online = build_network(device)
    target = build_network(device)
    behavior = build_network(device)
    target.load_state_dict(online.state_dict())
    behavior.load_state_dict(online.state_dict())
    target.eval()
    behavior.eval()
    optimizer = make_optimizer(online)

    buffer_size = int(config.SMOKE["buffer_size"] if smoke else config.DQN["buffer_size"])
    batch_size = int(config.SMOKE["batch_size"] if smoke else config.DQN["batch_size"])
    initial_warmup = int(config.SMOKE["learning_starts"] if smoke else config.DQN["learning_starts"])
    transition_warmup = int(
        config.SMOKE["stage_replay_warmup_steps"] if smoke
        else config.DOMAIN_RANDOMIZATION["stage_replay_warmup_steps"]
    )
    eval_freq = int(config.SMOKE["eval_freq"] if smoke else config.EVAL["eval_freq"])
    eval_episodes = int(config.SMOKE["n_eval_episodes"] if smoke else config.EVAL["n_eval_episodes"])
    eval_max_steps = int(config.SMOKE["max_eval_policy_steps"] if smoke else config.EVAL["max_eval_policy_steps"])
    checkpoint_freq = int(config.SMOKE["checkpoint_freq"] if smoke else config.EVAL["checkpoint_freq"])

    replay = ReplayBuffer(buffer_size, int(config.ENV["observation_dim"]))
    if replay.states.shape[1] != 6:
        raise AssertionError(f"Replay observation dimension must be 6, got {replay.states.shape}")

    episode_csv = run_dir / "episode_history.csv"
    eval_csv = run_dir / "eval_history.csv"
    dr_csv = run_dir / "dr_audit.csv"
    _write_csv_header(episode_csv, [
        "episode", "global_step", "stage_index", "stage_name", "level", "epsilon",
        "reward_total", "reward_per_step", "length", "maintain_max_stable", "stable_success",
        "capture", "mean_abs_alpha", "mean_abs_alpha_dot", "mean_abs_pwm",
    ])
    _write_csv_header(eval_csv, [
        "global_step", "phase", "eval_level", "mean_reward_per_step", "mean_episode_reward",
        "stable_success_rate", "capture_rate", "mean_maintain_max_stable", "mean_length",
    ])
    _write_csv_header(dr_csv, ["global_step", "stage_index", "stage_name", "level", "domain_json"])

    stats: Dict[str, Any] = {
        "episode_steps": [], "episode_reward_per_step": [], "episode_reward_total": [],
        "episode_lengths": [], "maintain_max_stable": [], "eval_history": [],
        "train_losses": [], "stop_reason": "not_finished",
    }
    best_nominal: Dict[str, Any] | None = None
    best_randomized: Dict[str, Any] | None = None
    randomized_no_improve = 0
    global_step = 0
    episode = 0
    next_eval = eval_freq
    next_checkpoint = checkpoint_freq
    next_progress = int(config.PANEL["progress_update_freq"])
    nominal_success_saved = False
    stopped_early = False
    start_time = time.perf_counter()

    emit_event(
        "training_started", run_dir=str(run_dir.resolve()), total_timesteps=total_steps,
        observation_shape=[6], replay_observation_shape=list(replay.states.shape),
        device=str(device), matlab_aligned=True,
    )
    print("=" * 100)
    print("[DQN]")
    print(f"run_dir       : {run_dir.resolve()}")
    print(f"device        : {device}")
    print(f"observation   : {online.fc1.in_features} -> {tuple(config.DQN['net_arch'])} -> {online.fc3.out_features}")
    print(f"replay shape  : {replay.states.shape}")
    print(f"learning rate : {config.DQN['learning_rate']}")
    print(f"vanilla DQN   : {not config.DQN['use_double_dqn']}")
    print(f"behavior sync : every {config.DQN['behavior_snapshot_interval']} steps")
    print("=" * 100, flush=True)

    def save_current(path: Path, stage_idx: int, stage_name: str, level: float) -> Path:
        payload = _make_payload(
            online, target, behavior, optimizer, replay,
            global_step=global_step, episode=episode, stage_index=stage_idx,
            stage_name=stage_name, stage_level=level, stats=stats,
        )
        return save_checkpoint(path, payload)

    def run_eval(stage_idx: int, stage_name: str, stage_level: float, force_level: float | None = None) -> Dict[str, Any]:
        nonlocal best_nominal, best_randomized, randomized_no_improve
        phase = "nominal" if stage_idx == 0 else "randomized"
        eval_level = (
            float(config.EVAL["nominal_eval_randomization_level"])
            if phase == "nominal"
            else float(config.EVAL["randomized_eval_randomization_level"])
        )
        if force_level is not None:
            eval_level = float(force_level)
        metrics = evaluate_network(
            online, device=device, randomization_level=eval_level,
            episodes=eval_episodes, max_steps=eval_max_steps, seed=seed + 100_000 + global_step,
        )
        metrics.update({
            "step": global_step, "phase": phase, "stage_name": stage_name,
            "training_level": stage_level, "eval_randomization_level": eval_level,
        })
        stats["eval_history"].append(metrics)
        dump_json(run_dir / "eval_history.json", stats["eval_history"])
        _append_csv(eval_csv, [
            global_step, phase, eval_level, metrics["mean_reward_per_step"],
            metrics["mean_episode_reward"], metrics["stable_success_rate"], metrics["capture_rate"],
            metrics["mean_maintain_max_stable"], metrics["mean_length"],
        ])

        if phase == "nominal":
            improved = _is_better(metrics, best_nominal)
            if improved:
                best_nominal = dict(metrics)
                save_current(nominal_best_dir / "best_model.pt", stage_idx, stage_name, stage_level)
                export_c_header(online, nominal_best_dir / "model_weights.h")
                dump_json(nominal_best_dir / "best_metrics.json", metrics)
        else:
            improved = _is_better(metrics, best_randomized)
            if improved:
                best_randomized = dict(metrics)
                randomized_no_improve = 0
                save_current(randomized_best_dir / "best_model.pt", stage_idx, stage_name, stage_level)
                export_c_header(online, randomized_best_dir / "model_weights.h")
                dump_json(randomized_best_dir / "best_metrics.json", metrics)
            else:
                randomized_no_improve += 1

        emit_event(
            "evaluation", step=global_step, phase=phase,
            eval_randomization_level=eval_level,
            mean_reward=float(metrics["mean_reward_per_step"]),
            mean_episode_reward=float(metrics["mean_episode_reward"]),
            stable_success_rate=float(metrics["stable_success_rate"]),
            capture_rate=float(metrics["capture_rate"]),
            mean_maintain_max_stable=float(metrics["mean_maintain_max_stable"]),
            improved=bool(improved), no_improve_evals=randomized_no_improve,
        )
        print(
            f"[EVAL] step={global_step:,} phase={phase} train_level={stage_level:.2f} "
            f"eval_level={eval_level:.2f} reward/step={metrics['mean_reward_per_step']:.5f} "
            f"success={100*metrics['stable_success_rate']:.1f}% "
            f"maintain={metrics['mean_maintain_max_stable']:.1f}",
            flush=True,
        )
        return metrics

    for stage_idx, (stage_name, level, planned_steps) in enumerate(zip(names, levels, stage_steps)):
        stage_number = stage_idx + 1
        if stage_idx > 0:
            replay_before = replay.size
            if bool(config.DOMAIN_RANDOMIZATION["clear_replay_between_stages"]):
                replay.reset()
            optimizer_reset = False
            if bool(config.DOMAIN_RANDOMIZATION["reset_optimizer_between_stages"]):
                optimizer = make_optimizer(online)
                optimizer_reset = True
            target_synced = False
            if bool(config.DOMAIN_RANDOMIZATION["sync_target_between_stages"]):
                target.load_state_dict(online.state_dict())
                target_synced = True
            behavior_synced = False
            if bool(config.DOMAIN_RANDOMIZATION["sync_behavior_snapshot_between_stages"]):
                behavior.load_state_dict(online.state_dict())
                behavior_synced = True
            emit_event(
                "stage_transition", stage=stage_number, stage_name=stage_name, level=level,
                replay_size_before=replay_before, replay_size_after=replay.size,
                replay_cleared=(replay.size == 0 and replay_before > 0),
                optimizer_reset=optimizer_reset, target_synced=target_synced,
                behavior_synced=behavior_synced,
            )

        emit_event(
            "training_stage", stage_index=stage_number, stage_count=len(stage_steps),
            stage_name=stage_name, level=level, stage_steps=planned_steps,
            replay_cleared=stage_idx > 0 and replay.size == 0,
        )
        stage_start = global_step
        stage_end = stage_start + planned_steps
        stage_warmup = initial_warmup if stage_idx == 0 else transition_warmup
        env = MatlabAlignedRIPEnv(randomization_level=level, seed=seed + stage_idx * 10_000)

        while global_step < stage_end:
            episode += 1
            obs, reset_info = env.reset()
            if bool(config.PANEL.get("save_dr_audit_csv", True)):
                _append_csv(dr_csv, [global_step, stage_number, stage_name, level, reset_info["domain"]])
            rewards: list[float] = []
            infos: list[Dict[str, Any]] = []
            episode_start_global = global_step
            last_epsilon = _stage_epsilon(global_step, global_step - stage_start, stage_idx)

            while global_step < stage_end:
                global_step += 1
                stage_step = global_step - stage_start
                epsilon = _stage_epsilon(global_step, stage_step, stage_idx)
                last_epsilon = epsilon
                if rng.random() < epsilon:
                    action = int(rng.integers(0, len(config.ENV["discrete_actions"])))
                else:
                    action = greedy_action(behavior, obs, device)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done_for_bootstrap = bool(terminated)  # Time-limit truncation is not a physical terminal state.
                replay.add(obs, action, reward, next_obs, done_for_bootstrap)
                obs = next_obs
                rewards.append(float(reward))
                infos.append(info)

                if replay.size >= stage_warmup and global_step % int(config.DQN["train_freq"]) == 0:
                    for _ in range(int(config.DQN["gradient_steps"])):
                        update_metrics = dqn_update(
                            online, target, optimizer, replay, rng, device, batch_size,
                        )
                        stats["train_losses"].append({"step": global_step, **update_metrics})
                    if len(stats["train_losses"]) > 5000:
                        stats["train_losses"] = stats["train_losses"][-5000:]

                if global_step % int(config.DQN["target_update_interval"]) == 0:
                    target.load_state_dict(online.state_dict())
                if global_step % int(config.DQN["behavior_snapshot_interval"]) == 0:
                    behavior.load_state_dict(online.state_dict())

                if global_step >= next_eval:
                    run_eval(stage_idx, stage_name, level)
                    while next_eval <= global_step:
                        next_eval += eval_freq

                if global_step >= next_checkpoint:
                    save_current(checkpoints_dir / f"model_{global_step:09d}.pt", stage_idx, stage_name, level)
                    while next_checkpoint <= global_step:
                        next_checkpoint += checkpoint_freq

                if global_step >= next_progress:
                    emit_event(
                        "progress", step=global_step, total=total_steps, stage_index=stage_number,
                        stage_name=stage_name, level=level, replay_size=replay.size,
                        epsilon=epsilon, elapsed_s=time.perf_counter() - start_time,
                    )
                    while next_progress <= global_step:
                        next_progress += int(config.PANEL["progress_update_freq"])

                if terminated or truncated:
                    break

            summary = episode_summary(rewards, infos, global_step - episode_start_global)
            stats["episode_steps"].append(global_step)
            stats["episode_reward_per_step"].append(summary["reward_per_step"])
            stats["episode_reward_total"].append(summary["reward_total"])
            stats["episode_lengths"].append(summary["length"])
            stats["maintain_max_stable"].append(summary["maintain_max_stable"])
            _append_csv(episode_csv, [
                episode, global_step, stage_number, stage_name, level, last_epsilon,
                summary["reward_total"], summary["reward_per_step"], summary["length"],
                summary["maintain_max_stable"], int(summary["stable_success"]), int(summary["capture"]),
                summary["mean_abs_alpha"], summary["mean_abs_alpha_dot"], summary["mean_abs_pwm"],
            ])
            emit_event(
                "episode", step=global_step, episode=episode,
                reward=float(summary["reward_per_step"]), reward_total=float(summary["reward_total"]),
                length=int(summary["length"]), maintain_max_stable=int(summary["maintain_max_stable"]),
                stable_success=bool(summary["stable_success"]), epsilon=float(last_epsilon),
                stage_name=stage_name, level=level,
            )

            window = int(config.DQN["stop_avg_window"])
            if len(stats["maintain_max_stable"]) >= window:
                recent = float(np.mean(stats["maintain_max_stable"][-window:]))
                if (
                    stage_idx == 0
                    and recent > float(config.DQN["stop_avg_stable_steps_threshold"])
                    and not nominal_success_saved
                ):
                    nominal_success_saved = True
                    success_path = save_current(
                        recovery_dir / "matlab_success_snapshot.pt", stage_idx, stage_name, level,
                    )
                    export_c_header(online, recovery_dir / "matlab_success_model_weights.h")
                    dump_json(recovery_dir / "matlab_success.json", {
                        "step": global_step, "episode": episode, "recent_mean_maintain": recent,
                        "model": str(success_path),
                    })
                    emit_event("nominal_success", step=global_step, recent_mean_maintain=recent, model=str(success_path))
                    if bool(config.DQN.get("allow_early_nominal_stage_transition", False)):
                        global_step = stage_end
                        stats["stop_reason"] = "nominal_success_early_stage_transition"
                        break

            # Randomized phase early stopping, never active in the first 2M stage.
            if (
                stage_idx > 0
                and bool(config.EVAL["early_stop_enabled"])
                and global_step >= int(total_steps * float(config.EVAL["early_stop_start_fraction"]))
                and best_randomized is not None
                and float(best_randomized["stable_success_rate"]) >= float(config.EVAL["early_stop_min_success_rate"])
                and randomized_no_improve >= int(config.EVAL["early_stop_patience_evals"])
            ):
                stopped_early = True
                stats["stop_reason"] = "randomized_no_improvement"
                reason = (
                    f"randomized best success={best_randomized['stable_success_rate']:.3f}; "
                    f"no improvement for {randomized_no_improve} evaluations"
                )
                dump_json(run_dir / "early_stop.json", {"step": global_step, "reason": reason, "best": best_randomized})
                emit_event("early_stop", step=global_step, reason=reason)
                break

        # Force an evaluation at every stage end, even if not aligned with eval_freq.
        if not stats["eval_history"] or int(stats["eval_history"][-1]["step"]) != global_step:
            run_eval(stage_idx, stage_name, level)

        if stage_idx == 0:
            nominal_last = save_current(recovery_dir / "nominal_2m_last.pt", stage_idx, stage_name, level)
            export_c_header(online, recovery_dir / "nominal_2m_last_model_weights.h")
            emit_event("nominal_recovery_saved", step=global_step, model=str(nominal_last))
        env.close()
        if stopped_early:
            break

    final_path = save_current(
        run_dir / "final_model.pt", min(len(names) - 1, stage_idx), names[min(len(names) - 1, stage_idx)], levels[min(len(levels) - 1, stage_idx)],
    )
    export_c_header(online, run_dir / "final_model_weights.h")

    nominal_path = nominal_best_dir / "best_model.pt"
    randomized_path = randomized_best_dir / "best_model.pt"
    threshold = float(config.EVAL["randomized_model_min_success_for_selection"])
    selected_source: Path
    selected_reason: str
    if randomized_path.is_file() and best_randomized is not None and float(best_randomized["stable_success_rate"]) >= threshold:
        selected_source = randomized_path
        selected_reason = "randomized_best_passed_success_threshold"
    elif nominal_path.is_file():
        selected_source = nominal_path
        selected_reason = "nominal_best_fallback"
    elif (recovery_dir / "nominal_2m_last.pt").is_file():
        selected_source = recovery_dir / "nominal_2m_last.pt"
        selected_reason = "nominal_2m_last_fallback"
    else:
        selected_source = final_path
        selected_reason = "final_fallback"

    selected_path = copy_model(selected_source, run_dir / "selected_best_model.pt")
    selected_loaded = torch.load(selected_path, map_location=device, weights_only=False)
    selected_net = build_network(device)
    selected_net.load_state_dict(selected_loaded["online_state_dict"])
    selected_net.eval()
    export_c_header(selected_net, deploy_dir / "model_weights.h")
    export_c_header(selected_net, deploy_dir / "model_weights_current.h")
    (deploy_dir / "active_variant.txt").write_text("current\n", encoding="utf-8")

    stats["stop_reason"] = stats["stop_reason"] if stats["stop_reason"] != "not_finished" else (
        "early_stop" if stopped_early else "max_steps_reached"
    )
    summary = {
        "run_dir": str(run_dir.resolve()),
        "global_step": global_step,
        "episodes": episode,
        "stop_reason": stats["stop_reason"],
        "observation_dim": 6,
        "replay_observation_shape": list(replay.states.shape),
        "selected_model": str(selected_path.resolve()),
        "selected_reason": selected_reason,
        "best_nominal": best_nominal,
        "best_randomized": best_randomized,
        "header": str((deploy_dir / "model_weights.h").resolve()),
        "device": str(device),
    }
    dump_json(run_dir / "training_summary.json", summary)
    emit_event("training_finished", **summary)
    print("=" * 100)
    print("Training finished")
    print(f"Run directory : {run_dir.resolve()}")
    print(f"Selected model: {selected_path.resolve()} ({selected_reason})")
    print(f"C header      : {(deploy_dir / 'model_weights.h').resolve()}")
    print("=" * 100, flush=True)
    return summary


if __name__ == "__main__":
    run_training(smoke=False)
