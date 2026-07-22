"""TD3 training panel aligned with the current DQN/PPO interface.

User-facing entry:
    python run.py

The panel keeps every original TD3 training/configuration setting intact.
Only the entry presentation, live summaries, and chart display are changed.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from config_editor import ConfigEditError, update_config_file

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.py"


def _get_path_value(module: Any, dotted: str) -> Any:
    parts = dotted.split(".")
    value = getattr(module, parts[0])
    for part in parts[1:]:
        value = value[part]
    return value


def _flatten(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(f"{prefix}.{key}", child)
    else:
        yield prefix, value


def _parse_text(text: str, original: Any) -> Any:
    text = text.strip()
    if isinstance(original, str):
        return text
    if isinstance(original, bool):
        lowered = text.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"需要布尔值，收到：{text}")
    try:
        value = ast.literal_eval(text)
    except Exception:
        if isinstance(original, float):
            value = float(text)
        elif isinstance(original, int):
            value = int(text)
        else:
            raise ValueError(f"无法解析：{text}")
    if isinstance(original, tuple) and isinstance(value, list):
        value = tuple(value)
    return value


class TrainingPanel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RIP TD3 Training Panel")
        self.geometry(str(config.PANEL.get("window_geometry", "1420x900")))
        self.minsize(1120, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.proc: subprocess.Popen | None = None
        self.output_queue: queue.Queue = queue.Queue()
        self.active_stage = "idle"
        self.workflow_locked = False
        self.workflow_smoke = False
        self.last_run_dir = ""
        self.current_values: Dict[str, Any] = {}
        self.advanced_values: Dict[str, Any] = {}
        self.entries: Dict[str, tk.StringVar] = {}
        self.config_widgets: list[tk.Widget] = []
        self.pending_stage_event: Dict[str, Any] | None = None
        self.episode_steps: list[int] = []
        self.episode_reward_per_step: list[float] = []
        self.episode_reward_ma30: list[float] = []
        self.episode_completion_ma30: list[float] = []
        self.eval_data: Dict[str, Dict[str, list[float]]] = {
            "nominal": {
                "steps": [], "reward_per_step": [], "capture": [],
                "success": [], "alpha": [], "level": [],
            },
            "randomized": {
                "steps": [], "reward_per_step": [], "capture": [],
                "success": [], "alpha": [], "level": [],
            },
        }
        self.last_train_summary = "Waiting for completed training episodes."
        self._episode_lengths: list[int] = []
        self.last_eval_summary = "Waiting for deterministic evaluation."
        self._plot_pending = False

        self._build_ui()
        self.reload_config()
        self.after(100, self._poll_output)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(10, 8, 10, 4))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Single entry point: python run.py",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text=(
                "7-D input · Actor 7→64→64→1 · twin Critics (64, 64) · "
                "first 2M steps nominal, then DR 0.10 → 0.30 → 0.50"
            ),
            foreground="#444",
        ).pack(side="left", padx=20)

        toolbar = ttk.Frame(self, padding=(8, 2, 8, 6))
        toolbar.pack(fill="x")
        self.save_button = ttk.Button(toolbar, text="Save to config.py", command=self.save_basic)
        self.save_button.pack(side="left", padx=3)
        self.reload_button = ttk.Button(toolbar, text="Reload", command=self.reload_config)
        self.reload_button.pack(side="left", padx=3)
        self.start_button = ttk.Button(toolbar, text="Start Full Training", command=lambda: self.start_training(False))
        self.start_button.pack(side="left", padx=3)
        self.smoke_button = ttk.Button(toolbar, text="Run Full Smoke Test", command=lambda: self.start_training(True))
        self.smoke_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(toolbar, text="Emergency Stop", command=self.stop_process, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        ttk.Button(toolbar, text="Open Runs Directory", command=self.open_runs).pack(side="right", padx=3)
        self.config_widgets.extend([self.save_button, self.reload_button, self.start_button, self.smoke_button])

        progress_frame = ttk.Frame(self, padding=(8, 0, 8, 6))
        progress_frame.pack(fill="x")
        self.progress = ttk.Progressbar(progress_frame, maximum=100.0)
        self.progress.pack(side="left", fill="x", expand=True)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var, width=54).pack(side="left", padx=(10, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.basic_tab = ttk.Frame(self.notebook)
        self.advanced_tab = ttk.Frame(self.notebook)
        self.workflow_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.basic_tab, text="Common Parameters")
        self.notebook.add(self.advanced_tab, text="All Config Parameters")
        self.notebook.add(self.workflow_tab, text="Training, Testing, and Curves")
        self.notebook.add(self.log_tab, text="Full Log")

        self._build_basic_tab()
        self._build_advanced_tab()
        self._build_workflow_tab()
        self._build_log_tab()

    def _field_groups(self):
        return [
            ("环境与原版7维观测", [
                ("物理步长", "ENV.physical_dt"),
                ("动作重复次数", "ENV.action_repeat"),
                ("episode 最大物理步数", "ENV.max_physical_steps"),
                ("连续 PWM 上限", "ENV.pwm_limit"),
                ("观测类型", "ENV.observation_type"),
                ("观测维度", "ENV.observation_dim"),
                ("速度裁剪", "ENV.clip_velocity_in_obs"),
                ("起始 theta 均值", "ENV.init_theta_mean_deg"),
                ("起始 theta 标准差", "ENV.init_theta_std_deg"),
                ("起始 theta_dot 标准差", "ENV.init_theta_dot_std"),
                ("起始 alpha 均值", "ENV.init_alpha_mean_deg"),
                ("起始 alpha 标准差", "ENV.init_alpha_std_deg"),
                ("起始 alpha_dot 标准差", "ENV.init_alpha_dot_std"),
                ("theta 速度上限", "ENV.theta_dot_limit"),
                ("alpha 速度上限", "ENV.alpha_dot_limit"),
                ("LPF 速度估计", "ENV.use_lpf_velocity"),
                ("LPF 系数", "ENV.velocity_lpf"),
            ]),
            ("TD3 Actor/Critic", [
                ("总训练步数", "TD3.total_timesteps"),
                ("并行环境数", "TD3.n_envs"),
                ("Replay buffer", "TD3.buffer_size"),
                ("开始学习步数", "TD3.learning_starts"),
                ("Batch size", "TD3.batch_size"),
                ("Actor 学习率", "TD3.actor_learning_rate"),
                ("Critic 学习率", "TD3.critic_learning_rate"),
                ("Gamma", "TD3.gamma"),
                ("Tau", "TD3.tau"),
                ("Train freq", "TD3.train_freq"),
                ("Gradient steps", "TD3.gradient_steps"),
                ("Policy delay", "TD3.policy_delay"),
                ("Target policy noise", "TD3.target_policy_noise"),
                ("Target noise clip", "TD3.target_noise_clip"),
                ("Actor 梯度裁剪", "TD3.actor_grad_clip"),
                ("Critic 梯度裁剪", "TD3.critic_grad_clip"),
                ("探索动作噪声 sigma", "TD3.action_noise_sigma"),
                ("Actor 网络结构", "TD3.net_arch_pi"),
                ("Critic 网络结构", "TD3.net_arch_qf"),
                ("激活函数", "TD3.activation_fn_name"),
            ]),
            ("固定阶段随机化", [
                ("启用随机化", "DOMAIN_RANDOMIZATION.enabled"),
                ("训练阶段名称", "DOMAIN_RANDOMIZATION.training_stage_names"),
                ("训练阶段 levels", "DOMAIN_RANDOMIZATION.training_stage_levels"),
                ("训练阶段显式步数", "DOMAIN_RANDOMIZATION.training_stage_steps"),
                ("训练阶段 fractions（说明）", "DOMAIN_RANDOMIZATION.training_stage_fractions"),
                ("Nominal 原版对齐步数", "DOMAIN_RANDOMIZATION.nominal_recovery_steps"),
                ("阶段间清空 replay", "DOMAIN_RANDOMIZATION.clear_replay_between_stages"),
                ("阶段间重置 optimizer", "DOMAIN_RANDOMIZATION.reset_optimizer_between_stages"),
                ("阶段间同步 actor/critic target", "DOMAIN_RANDOMIZATION.sync_target_between_stages"),
                ("阶段间重置动作噪声", "DOMAIN_RANDOMIZATION.reset_action_noise_between_stages"),
                ("新阶段 replay warm-up", "DOMAIN_RANDOMIZATION.stage_replay_warmup_steps"),
                ("执行器时间常数范围", "DOMAIN_RANDOMIZATION.actuator_tau_range"),
                ("动作延迟步数范围", "DOMAIN_RANDOMIZATION.action_delay_steps_range"),
                ("PWM 上限倍率范围", "DOMAIN_RANDOMIZATION.pwm_limit_scale_range"),
                ("PWM 增益范围", "DOMAIN_RANDOMIZATION.pwm_gain_range"),
                ("PWM 偏置范围", "DOMAIN_RANDOMIZATION.pwm_bias_range"),
                ("PWM 死区范围", "DOMAIN_RANDOMIZATION.pwm_deadzone_range"),
                ("PWM 噪声范围", "DOMAIN_RANDOMIZATION.pwm_noise_sigma_range"),
                ("theta 噪声范围", "DOMAIN_RANDOMIZATION.theta_sigma_range"),
                ("alpha 噪声范围", "DOMAIN_RANDOMIZATION.alpha_sigma_range"),
                ("theta_dot 噪声范围", "DOMAIN_RANDOMIZATION.theta_dot_sigma_range"),
                ("alpha_dot 噪声范围", "DOMAIN_RANDOMIZATION.alpha_dot_sigma_range"),
                ("编码器量化范围", "DOMAIN_RANDOMIZATION.encoder_quantization_rad_range"),
                ("LPF 启用概率", "DOMAIN_RANDOMIZATION.use_lpf_velocity_probability"),
                ("LPF 系数范围", "DOMAIN_RANDOMIZATION.velocity_lpf_range"),
            ]),
            ("物理参数随机化范围", [
                ("重力倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.g_scale"),
                ("m1 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.m1_scale"),
                ("m2 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.m2_scale"),
                ("l1 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.l1_scale"),
                ("l1cg 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.l1cg_scale"),
                ("l2cg 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.l2cg_scale"),
                ("I1z 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.I1z_scale"),
                ("I2x 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.I2x_scale"),
                ("I2y 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.I2y_scale"),
                ("I2z 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.I2z_scale"),
                ("theta 摩擦倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.c_theta_scale"),
                ("alpha 摩擦倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.c_alpha_scale"),
                ("k_t 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.k_t_scale"),
                ("k_b 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.k_b_scale"),
                ("k_u 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.k_u_scale"),
                ("R 倍率", "DOMAIN_RANDOMIZATION.param_scale_ranges.R_scale"),
            ]),
            ("原版 TD3 奖励", [
                ("直立门控奖励", "REWARD.base_reward"),
                ("theta² 权重", "REWARD.a_theta"),
                ("alpha² 权重", "REWARD.a_alpha"),
                ("theta_dot² 权重", "REWARD.a_theta_dot"),
                ("alpha_dot² 权重", "REWARD.a_alpha_dot"),
                ("动作 u² 权重", "REWARD.a_u"),
                ("动作变化 du² 权重", "REWARD.a_du"),
                ("门控 theta 上限(rad)", "REWARD.gate_theta_max_rad"),
                ("门控 theta_dot 上限", "REWARD.gate_theta_dot_max"),
                ("门控 alpha 上限(deg)", "REWARD.gate_alpha_max_deg"),
                ("门控 alpha_dot 上限", "REWARD.gate_alpha_dot_max"),
            ]),
            ("评估、Best 与早停", [
                ("评估间隔", "EVAL.eval_freq"),
                ("评估回合数", "EVAL.n_eval_episodes"),
                ("评估最大步数", "EVAL.max_eval_policy_steps"),
                ("Checkpoint 间隔", "EVAL.checkpoint_freq"),
                ("保存 best model", "EVAL.save_best_model"),
                ("Nominal 评估 level", "EVAL.nominal_eval_randomization_level"),
                ("随机化评估 level", "EVAL.eval_randomization_level"),
                ("选择随机模型最低成功率", "EVAL.selected_model_min_success_rate"),
                ("捕获角度", "EVAL.capture_angle_deg"),
                ("稳定速度阈值", "EVAL.stable_alpha_dot_max"),
                ("持续稳定时间", "EVAL.stable_hold_seconds"),
                ("启用早停", "EVAL.early_stop_enabled"),
                ("早停起始比例", "EVAL.early_stop_start_fraction"),
                ("早停耐心次数", "EVAL.early_stop_patience_evals"),
                ("早停最低成功率", "EVAL.early_stop_min_success_rate"),
                ("Reward 最小改善", "EVAL.early_stop_reward_min_delta"),
            ]),
            ("Current Actor 蒸馏", [
                ("学生隐藏层", "DISTILL.student_hidden_sizes"),
                ("DAgger 轮数", "DISTILL.dagger_iterations"),
                ("每轮采样步数", "DISTILL.collect_steps_per_iter"),
                ("最大数据集", "DISTILL.max_dataset_size"),
                ("学生动作采样概率", "DISTILL.student_action_probability"),
                ("采样随机化 levels", "DISTILL.collect_randomization_levels"),
                ("每轮 epochs", "DISTILL.epochs_per_iter"),
                ("蒸馏 batch size", "DISTILL.batch_size"),
                ("蒸馏学习率", "DISTILL.learning_rate"),
                ("Weight decay", "DISTILL.weight_decay"),
                ("蒸馏评估 level", "DISTILL.eval_randomization_level"),
                ("蒸馏评估回合", "DISTILL.eval_episodes"),
            ]),
            ("30秒测试与运行", [
                ("测试时长", "TEST.duration_seconds"),
                ("测试随机化 level", "TEST.randomization_level"),
                ("测试随机种子", "TEST.seed"),
                ("运行随机种子", "RUN.seed"),
                ("Torch 线程数", "RUN.torch_num_threads"),
                ("设备", "RUN.device"),
                ("Vec env 类型", "RUN.vec_env_type"),
                ("运行目录", "RUN.root_log_dir"),
            ]),
        ]

    def _build_basic_tab(self) -> None:
        outer = ttk.Frame(self.basic_tab)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=8)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        row = 0
        for group_name, fields in self._field_groups():
            frame = ttk.LabelFrame(inner, text=group_name, padding=8)
            frame.grid(row=row, column=0, sticky="ew", pady=5)
            frame.columnconfigure(1, weight=1)
            row += 1
            for index, (label, path) in enumerate(fields):
                ttk.Label(frame, text=label, width=30).grid(
                    row=index, column=0, sticky="w", padx=(0, 8), pady=2
                )
                var = tk.StringVar()
                entry = ttk.Entry(frame, textvariable=var, width=58)
                entry.grid(row=index, column=1, sticky="ew", pady=2)
                ttk.Label(frame, text=path, foreground="#666").grid(
                    row=index, column=2, sticky="w", padx=(8, 0)
                )
                self.entries[path] = var
                self.config_widgets.append(entry)
        inner.columnconfigure(0, weight=1)

    def _build_advanced_tab(self) -> None:
        top = ttk.Frame(self.advanced_tab, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="此页动态列出 config.py 的全部字典字段，所有参数均可选中、编辑并写回。", wraplength=950).pack(side="left")
        self.apply_advanced_button = ttk.Button(top, text="应用选中值", command=self.apply_advanced)
        self.apply_advanced_button.pack(side="right")
        self.config_widgets.append(self.apply_advanced_button)
        columns = ("path", "value", "type")
        self.tree = ttk.Treeview(self.advanced_tab, columns=columns, show="headings")
        self.tree.heading("path", text="Config path")
        self.tree.heading("value", text="Value")
        self.tree.heading("type", text="Type")
        self.tree.column("path", width=430, anchor="w")
        self.tree.column("value", width=650, anchor="w")
        self.tree.column("type", width=100, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8)
        self.tree.bind("<<TreeviewSelect>>", self._advanced_selected)
        edit = ttk.Frame(self.advanced_tab, padding=8)
        edit.pack(fill="x")
        self.advanced_path_var = tk.StringVar()
        self.advanced_edit_var = tk.StringVar()
        ttk.Label(edit, textvariable=self.advanced_path_var, width=60).pack(side="left")
        self.advanced_entry = ttk.Entry(edit, textvariable=self.advanced_edit_var)
        self.advanced_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.config_widgets.extend([self.tree, self.advanced_entry])

    def _build_workflow_tab(self) -> None:
        summary = ttk.LabelFrame(self.workflow_tab, text="Workflow and Current Status", padding=10)
        summary.pack(fill="x", padx=8, pady=8)
        summary.columnconfigure(0, weight=1)

        self.run_dir_text = tk.StringVar(value="run_dir: waiting")
        self.progress_text = tk.StringVar(value=self.last_train_summary)
        self.stage_text = tk.StringVar(value="Stage: -- · level=-- · steps=--")
        self.eval_text = tk.StringVar(value=self.last_eval_summary)
        self.network_text = tk.StringVar(
            value="7-D input · Actor 7→64→64→1 · twin Critics 8→64→64→1 · direct TD3 model"
        )
        ttk.Label(summary, textvariable=self.network_text, font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=8, sticky="w"
        )
        ttk.Label(summary, textvariable=self.run_dir_text).grid(row=1, column=0, columnspan=8, sticky="w", pady=(4, 0))
        ttk.Label(summary, textvariable=self.progress_text).grid(row=2, column=0, columnspan=8, sticky="w", pady=(2, 0))
        ttk.Label(summary, textvariable=self.eval_text).grid(row=3, column=0, columnspan=8, sticky="w", pady=(2, 0))
        ttk.Label(summary, textvariable=self.stage_text).grid(row=4, column=0, columnspan=8, sticky="w", pady=(2, 8))

        self.distill_button = ttk.Button(
            summary, text="Start Current-Actor Distillation", command=self.start_distillation, state="disabled"
        )
        self.distill_button.grid(row=5, column=0, padx=(0, 8), sticky="w")
        self.test_button = ttk.Button(
            summary, text="Start 30-Second Current Test", command=self.start_test, state="disabled"
        )
        self.test_button.grid(row=5, column=1, padx=8, sticky="w")

        self.figure = Figure(figsize=(11, 7), dpi=100)
        self.ax_reward = self.figure.add_subplot(311)
        self.ax_eval = self.figure.add_subplot(312)
        self.ax_success = self.figure.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.workflow_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._redraw_plot()

    def _build_log_tab(self) -> None:
        self.log_text = scrolledtext.ScrolledText(self.log_tab, wrap="word", font=("Menlo", 10))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def reload_config(self) -> None:
        global config
        config = importlib.reload(config)
        self.current_values.clear()
        for path, var in self.entries.items():
            try:
                value = _get_path_value(config, path)
            except Exception:
                continue
            self.current_values[path] = value
            var.set(value if isinstance(value, str) else repr(value))
        self._refresh_advanced()
        self.status_var.set("config.py reloaded")

    def save_basic(self, quiet: bool = False) -> bool:
        updates: Dict[str, Any] = {}
        try:
            for path, var in self.entries.items():
                if path in self.current_values:
                    updates[path] = _parse_text(var.get(), self.current_values[path])
            backup = update_config_file(CONFIG_PATH, updates)
            self.reload_config()
            if not quiet:
                messagebox.showinfo("Saved", f"config.py updated\nBackup: {backup.name}")
            return True
        except (ValueError, ConfigEditError, OSError) as exc:
            messagebox.showerror("保存失败", str(exc))
            return False

    def _refresh_advanced(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.advanced_values.clear()
        sections = ["RUN", "ENV", "PHYSICAL_PARAMS", "DOMAIN_RANDOMIZATION", "TD3", "REWARD", "EVAL", "DISTILL", "TEST", "SMOKE", "PANEL"]
        for section in sections:
            if not hasattr(config, section):
                continue
            for path, value in _flatten(section, getattr(config, section)):
                self.advanced_values[path] = value
                display = value if isinstance(value, str) else repr(value)
                self.tree.insert("", "end", iid=path, values=(path, display, type(value).__name__))

    def _advanced_selected(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        path = selected[0]
        value = self.advanced_values[path]
        self.advanced_path_var.set(path)
        self.advanced_edit_var.set(value if isinstance(value, str) else repr(value))

    def apply_advanced(self) -> None:
        path = self.advanced_path_var.get()
        if not path:
            return
        try:
            value = _parse_text(self.advanced_edit_var.get(), self.advanced_values[path])
            backup = update_config_file(CONFIG_PATH, {path: value})
            self.reload_config()
            messagebox.showinfo("已写入", f"{path} 已写入 config.py\n备份：{backup.name}")
        except (ValueError, ConfigEditError, OSError) as exc:
            messagebox.showerror("保存失败", str(exc))

    def _set_config_locked(self, locked: bool) -> None:
        for widget in self.config_widgets:
            try:
                if isinstance(widget, ttk.Treeview):
                    continue
                widget.configure(state="disabled" if locked else "normal")
            except tk.TclError:
                pass

    def start_training(self, smoke: bool) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Already running", "A process is already running.")
            return
        if bool(config.PANEL.get("auto_save_before_run", True)) and not self.save_basic(quiet=True):
            return
        self.workflow_locked = True
        self.workflow_smoke = bool(smoke)
        self._set_config_locked(True)
        self.last_run_dir = ""
        self.episode_steps.clear()
        self.episode_reward_per_step.clear()
        self.episode_reward_ma30.clear()
        self.episode_completion_ma30.clear()
        self._episode_lengths.clear()
        for data in self.eval_data.values():
            for values in data.values():
                values.clear()
        self.last_train_summary = "Waiting for completed training episodes."
        self.last_eval_summary = "Waiting for deterministic evaluation."
        self.run_dir_text.set("run_dir: waiting")
        self.progress_text.set(self.last_train_summary)
        self.eval_text.set(self.last_eval_summary)
        self._redraw_plot()
        self.progress["value"] = 0.0
        self.distill_button.configure(state="disabled")
        self.test_button.configure(state="disabled")
        worker = "smoke" if smoke else "train"
        self._launch(
            [sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "--worker", worker],
            "training",
        )
        self.notebook.select(self.workflow_tab)

    def start_distillation(self, _target: str | None = None) -> None:
        if not self.last_run_dir:
            messagebox.showerror("缺少训练结果", "还没有可用的 run_dir")
            return
        self.distill_button.configure(state="disabled")
        self.test_button.configure(state="disabled")
        cmd = [sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "--worker", "distill", "--run-dir", self.last_run_dir, "--target", "current"]
        if self.workflow_smoke:
            cmd.append("--smoke")
        self._launch(cmd, "distillation")
        self.notebook.select(self.workflow_tab)

    def start_test(self, _variant: str | None = None) -> None:
        if not self.last_run_dir:
            messagebox.showerror("缺少结果", "还没有可用的 run_dir")
            return
        self.test_button.configure(state="disabled")
        cmd = [sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "--worker", "test", "--run-dir", self.last_run_dir, "--variant", "current"]
        if self.workflow_smoke:
            cmd.append("--smoke")
        self._launch(cmd, "test")
        self.notebook.select(self.workflow_tab)

    def _launch(self, cmd: list[str], stage: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("正在运行", "请等待当前进程完成")
            return
        self.active_stage = stage
        self.pending_stage_event = None
        self.stop_button.configure(state="normal")
        self.status_var.set(f"{stage} 运行中")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.log_text.insert("end", "\n$ " + " ".join(cmd) + "\n")
        try:
            self.proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            self.active_stage = "idle"
            return
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.output_queue.put(("line", line))
        self.output_queue.put(("done", self.proc.wait()))

    def _poll_output(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "line":
                    self._handle_line(payload)
                else:
                    self._handle_process_done(int(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def _handle_line(self, line: str) -> None:
        self.log_text.insert("end", line)
        self.log_text.see("end")
        prefix = "[PANEL_JSON] "
        if prefix not in line:
            return
        try:
            event = json.loads(line.split(prefix, 1)[1])
        except Exception:
            return

        event_type = str(event.get("event", ""))
        if event.get("run_dir"):
            self.last_run_dir = str(event["run_dir"])
            self.run_dir_text.set(f"run_dir: {self.last_run_dir}")

        if event_type == "episode":
            step = int(event.get("step", event.get("timesteps", 0)))
            reward = float(event.get("reward", 0.0))
            length = max(1, int(event.get("length", 1)))
            reward_per_step = reward / float(length)
            max_policy_steps = max(
                1,
                int(np.ceil(float(config.ENV["max_physical_steps"]) / max(int(config.ENV["action_repeat"]), 1))),
            )
            completed = 1.0 if length >= max_policy_steps else 0.0
            self.episode_steps.append(step)
            self.episode_reward_per_step.append(reward_per_step)
            window = min(30, len(self.episode_reward_per_step))
            self.episode_reward_ma30.append(float(np.mean(self.episode_reward_per_step[-window:])))
            completion_window = min(30, len(self.episode_completion_ma30) + 1)
            recent_completed = [
                1.0 if int(x) >= max_policy_steps else 0.0
                for x in getattr(self, "_episode_lengths", [])[-(completion_window - 1):]
            ] + [completed]
            if not hasattr(self, "_episode_lengths"):
                self._episode_lengths = []
            self._episode_lengths.append(length)
            self._episode_lengths = self._episode_lengths[-5000:]
            completion_rate = float(np.mean(recent_completed)) if recent_completed else completed
            self.episode_completion_ma30.append(completion_rate)
            limit = int(config.PANEL.get("episode_curve_max_points", 2000))
            self.episode_steps = self.episode_steps[-limit:]
            self.episode_reward_per_step = self.episode_reward_per_step[-limit:]
            self.episode_reward_ma30 = self.episode_reward_ma30[-limit:]
            self.episode_completion_ma30 = self.episode_completion_ma30[-limit:]
            self.last_train_summary = (
                f"Training episode: reward/step={reward_per_step:.5f}, "
                f"moving mean(30)={self.episode_reward_ma30[-1]:.5f}, "
                f"full-episode rate(30)={100.0 * completion_rate:.1f}%"
            )
            self.progress_text.set(self.last_train_summary)
            self._schedule_plot()

        elif event_type == "evaluation":
            step = int(event.get("step", 0))
            phase = str(event.get("phase", "nominal"))
            kind = "nominal" if phase == "nominal" else "randomized"
            mean_reward = float(event.get("mean_reward", 0.0))
            mean_length = max(1.0, float(event.get("mean_length", 1.0)))
            reward_per_step = float(event.get("reward_per_step", mean_reward / mean_length))
            success = float(event.get("stable_success_rate", 0.0))
            capture = float(event.get("capture_rate", 0.0))
            alpha = float(event.get("mean_abs_alpha", 0.0))
            level = float(event.get("eval_randomization_level", 0.0))
            data = self.eval_data[kind]
            for key, value in (
                ("steps", float(step)), ("reward_per_step", reward_per_step),
                ("capture", capture), ("success", success),
                ("alpha", alpha), ("level", level),
            ):
                data[key].append(value)
                if len(data[key]) > 1000:
                    del data[key][:-1000]
            total = max(1, int(config.TD3["total_timesteps"]))
            self.progress["value"] = min(100.0, 100.0 * step / total)
            parts = []
            for label in ("nominal", "randomized"):
                d = self.eval_data[label]
                if d["steps"]:
                    parts.append(
                        f"{label} r/step={d['reward_per_step'][-1]:.5f}, "
                        f"capture={100.0*d['capture'][-1]:.1f}%, "
                        f"stable={100.0*d['success'][-1]:.1f}%, "
                        f"|alpha|={d['alpha'][-1]:.4f}"
                    )
            self.last_eval_summary = "Deterministic eval: " + " | ".join(parts)
            self.eval_text.set(self.last_eval_summary)
            self.status_var.set(
                f"eval@{step:,} · {kind} level={level:.2f} · "
                f"stable={100.0*success:.1f}%"
            )
            self._schedule_plot()

        elif event_type == "training_stage":
            idx = int(event.get("stage_index", 0))
            count = int(event.get("stage_count", 0))
            level = float(event.get("level", 0.0))
            name = str(event.get("stage_name", ""))
            stage_steps = int(event.get("stage_steps", 0))
            self.stage_text.set(
                f"Stage {idx}/{count} · {name} · level={level:.2f} · "
                f"steps={stage_steps:,} · replay clear="
                f"{idx > 1 and bool(config.DOMAIN_RANDOMIZATION['clear_replay_between_stages'])}"
            )
        elif event_type == "stage_transition":
            self.stage_text.set(
                f"Stage transition · {event.get('stage_name', '')} · "
                f"level={float(event.get('level', 0.0)):.2f} · "
                f"replay {event.get('replay_size_before')}→{event.get('replay_size_after')} · "
                f"optimizer reset={event.get('optimizer_reset')} · target sync={event.get('target_synced')}"
            )
        elif event_type == "early_stop":
            reason = str(event.get("reason", "Early-stop condition met."))
            self.stage_text.set(f"Early stop: {reason}")
            self.status_var.set(f"Training early-stopped: {reason}")
        elif event_type == "distillation_iteration":
            self.progress_text.set(
                f"Current actor distillation: iter={event.get('iteration')} "
                f"dataset={event.get('dataset_size', event.get('dataset', ''))}"
            )
        elif event_type == "training_finished":
            self.pending_stage_event = event
            self.progress["value"] = 100.0
        elif event_type in {"distillation_finished", "test_finished"}:
            self.pending_stage_event = event

    def _handle_process_done(self, code: int) -> None:
        stage = self.active_stage
        self.proc = None
        self.stop_button.configure(state="disabled")
        self.active_stage = "idle"
        event = self.pending_stage_event
        self.pending_stage_event = None
        if code != 0:
            self.workflow_locked = False
            self._set_config_locked(False)
            self.status_var.set(f"{stage} 失败，退出码 {code}")
            messagebox.showerror("进程失败", f"{stage} 退出码：{code}\n请查看完整日志。")
            return
        if stage == "training":
            if self.workflow_smoke:
                self.workflow_locked = False
                self._set_config_locked(False)
                self.distill_button.configure(state="normal")
                self.test_button.configure(state="normal")
                self.status_var.set("完整 TD3 Smoke：训练、阶段切换、蒸馏和测试均已完成")
            else:
                self.distill_button.configure(state="normal")
                self.status_var.set("训练完成；best TD3 model 已选定，可开始 current Actor 蒸馏")
        elif stage == "distillation":
            self.test_button.configure(state="normal")
            self.status_var.set("current Actor 蒸馏完成，可开始 30 秒测试")
        elif stage == "test":
            self.workflow_locked = False
            self._set_config_locked(False)
            self.distill_button.configure(state="normal")
            self.test_button.configure(state="normal")
            result = (event or {}).get("result", {})
            self.status_var.set("TD3 训练、Actor 蒸馏、测试流程完成")
            messagebox.showinfo("测试完成", f"current 测试完成\n结果文件：{result.get('result_json', '见 run_dir')}\nTrace：{result.get('trace_csv', '见 run_dir')}")

    def _schedule_plot(self) -> None:
        if self._plot_pending:
            return
        self._plot_pending = True
        self.after(1000, self._redraw_plot)

    def _redraw_plot(self) -> None:
        self._plot_pending = False
        self.ax_reward.clear()
        self.ax_eval.clear()
        self.ax_success.clear()

        if self.episode_steps:
            self.ax_reward.plot(
                self.episode_steps,
                self.episode_reward_per_step,
                linewidth=0.8,
                alpha=0.45,
                label="episode",
            )
            self.ax_reward.plot(
                self.episode_steps,
                self.episode_reward_ma30,
                linewidth=1.8,
                label="moving mean (30)",
            )
        self.ax_reward.set_title("TD3 Episode Reward / Step")
        self.ax_reward.set_ylabel("reward / step")
        self.ax_reward.grid(True, alpha=0.3)
        if self.episode_steps:
            self.ax_reward.legend(loc="best")

        for kind, label in (("nominal", "nominal"), ("randomized", "randomized")):
            data = self.eval_data[kind]
            if data["steps"]:
                self.ax_eval.plot(
                    data["steps"], data["reward_per_step"], marker="o", markersize=3, label=label
                )
                self.ax_success.plot(
                    data["steps"], data["capture"], marker="o", markersize=3,
                    linestyle="--", label=f"{label} capture"
                )
                self.ax_success.plot(
                    data["steps"], data["success"], marker="s", markersize=3,
                    label=f"{label} stable"
                )
        self.ax_eval.set_title("Deterministic Evaluation Reward / Step")
        self.ax_eval.set_ylabel("reward / step")
        self.ax_eval.grid(True, alpha=0.3)
        if any(self.eval_data[k]["steps"] for k in self.eval_data):
            self.ax_eval.legend(loc="best")

        self.ax_success.set_title("Capture Rate and Stable Success Rate")
        self.ax_success.set_xlabel("timesteps")
        self.ax_success.set_ylabel("rate")
        self.ax_success.set_ylim(-0.03, 1.03)
        self.ax_success.grid(True, alpha=0.3)
        if any(self.eval_data[k]["steps"] for k in self.eval_data):
            self.ax_success.legend(loc="best", ncol=2)

        self.figure.tight_layout(pad=1.4)
        self.canvas.draw_idle()

    def stop_process(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if messagebox.askyesno("紧急停止", "确定终止当前进程吗？"):
            self.proc.terminate()
            self.workflow_locked = False
            self.status_var.set("已请求紧急停止")

    def open_runs(self) -> None:
        target = Path(self.last_run_dir) if self.last_run_dir else PROJECT_ROOT / str(config.RUN["root_log_dir"])
        target.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        elif os.name == "nt":
            os.startfile(str(target))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(target)])

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("不能关闭", "训练/蒸馏/测试正在运行。")
            return
        self.destroy()


def main() -> None:
    TrainingPanel().mainloop()


if __name__ == "__main__":
    main()
