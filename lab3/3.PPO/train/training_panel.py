"""Tkinter control panel for Stage-1 RIP training.

Primary entry:
    python run.py

``training_panel.py`` remains importable, but ``run.py`` is the only user-facing entry.
The panel edits ``config.py`` directly, starts ``run.py`` with the same Python
interpreter, displays stdout, and tracks the machine-readable ``[PROGRESS]``
lines emitted by the training callback.
"""

from __future__ import annotations

import ast
import importlib
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.py"
sys.path.insert(0, str(PROJECT_ROOT))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    Figure = None

import config
from config_editor import ConfigEditError, update_config_file


PROGRESS_RE = re.compile(
    r"\[PROGRESS\].*timesteps=(?P<steps>\d+).*total=(?P<total>\d+).*"
    r"percent=(?P<percent>[0-9.]+).*fps=(?P<fps>[0-9.]+).*eta_s=(?P<eta>[0-9.]+)"
)
FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
EVAL_RE = re.compile(r"\[EVAL\]\s+steps=(?P<steps>\d+).*random score=(?P<score>[-+0-9.eE]+), term=(?P<term>[-+0-9.eE]+)")
TRAIN_EPISODE_RE = re.compile(
    rf"\[TRAIN_EPISODE\]\s+timesteps=(?P<steps>\d+).*"
    rf"reward_per_step=(?P<reward>{FLOAT_PATTERN}).*"
    rf"moving_mean_30=(?P<moving>{FLOAT_PATTERN}).*"
    rf"success_rate_30=(?P<success>{FLOAT_PATTERN})"
)
EVAL_METRICS_RE = re.compile(
    rf"\[EVAL_METRICS\]\s+timesteps=(?P<steps>\d+)\s+eval_type=(?P<kind>nominal|randomized)\s+"
    rf"reward_per_step=(?P<reward>{FLOAT_PATTERN})\s+success_rate=(?P<success>{FLOAT_PATTERN}).*"
    rf"mean_abs_alpha=(?P<alpha>{FLOAT_PATTERN}).*terminated_rate=(?P<term>{FLOAT_PATTERN})"
)
RUN_DIR_RE = re.compile(r"\[RUN_DIR\]\s+path=(?P<path>.+)$")


def _parse_text(text: str, current_value: Any) -> Any:
    stripped = text.strip()
    if isinstance(current_value, bool):
        lowered = stripped.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError("Boolean value must be true/false")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(stripped.replace("_", ""))
    if isinstance(current_value, float):
        return float(stripped.replace("_", ""))
    if isinstance(current_value, str):
        return stripped
    try:
        return ast.literal_eval(stripped)
    except Exception as exc:
        raise ValueError(f"Use a valid Python literal, for example (0.5, 1.5): {exc}") from exc


def _get_path_value(module, dotted: str) -> Any:
    parts = dotted.split(".")
    value = getattr(module, parts[0])
    for part in parts[1:]:
        value = value[part]
    return value


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(f"{prefix}.{key}" if prefix else str(key), child)
    else:
        yield prefix, value


class TrainingPanel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.algorithm = str(getattr(config, "ALGORITHM_NAME", "PPO")).upper()
        self.algo_key = "PPO" if self.algorithm == "PPO" else "DQN"
        self.title(f"RIP Stage-1 {self.algorithm} Training Panel")
        geometry = str(getattr(config, "PANEL", {}).get("window_geometry", "1280x820"))
        self.geometry(geometry)
        self.minsize(1050, 700)

        self.proc: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.entries: Dict[str, tk.StringVar] = {}
        self.current_values: Dict[str, Any] = {}
        self.advanced_values: Dict[str, Any] = {}
        self.current_run_dir = ""
        self.train_steps: List[int] = []
        self.train_reward: List[float] = []
        self.train_reward_ma: List[float] = []
        self.train_success_ma: List[float] = []
        self.eval_data: Dict[str, Dict[str, List[float]]] = {
            "nominal": {"steps": [], "reward": [], "success": [], "alpha": []},
            "randomized": {"steps": [], "reward": [], "success": [], "alpha": []},
        }
        self.chart_dirty = True
        self.last_train_summary = "waiting for completed episodes"
        self.last_eval_summary = "waiting for deterministic evaluation"

        self._build_ui()
        self.reload_config()
        self.after(100, self._poll_output)
        self.after(1000, self._refresh_charts)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="保存到 config.py", command=self.save_basic).pack(side="left", padx=3)
        ttk.Button(toolbar, text="重新载入", command=self.reload_config).pack(side="left", padx=3)
        ttk.Button(toolbar, text="开始正式训练", command=lambda: self.start_process("train")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Smoke 测试", command=lambda: self.start_process("smoke")).pack(side="left", padx=3)
        ttk.Button(toolbar, text="停止", command=self.stop_process).pack(side="left", padx=3)
        ttk.Button(toolbar, text="打开 runs", command=self.open_runs).pack(side="left", padx=3)
        ttk.Label(toolbar, text=f"Algorithm: {self.algorithm}").pack(side="right", padx=8)

        status_frame = ttk.Frame(self, padding=(8, 0, 8, 8))
        status_frame.pack(fill="x")
        self.progress = ttk.Progressbar(status_frame, maximum=100.0)
        self.progress.pack(fill="x", side="left", expand=True)
        self.progress_label = ttk.Label(status_frame, text="idle", width=48)
        self.progress_label.pack(side="left", padx=(10, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.basic_tab = ttk.Frame(self.notebook)
        self.advanced_tab = ttk.Frame(self.notebook)
        self.progress_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        self.eval_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.basic_tab, text="常用参数")
        self.notebook.add(self.advanced_tab, text="全部 config 参数")
        self.notebook.add(self.progress_tab, text="训练、评估与曲线")
        self.notebook.add(self.log_tab, text="完整日志")
        self.notebook.add(self.eval_tab, text="模型评估")

        self._build_basic_tab()
        self._build_advanced_tab()
        self._build_progress_tab()
        self._build_log_tab()
        self._build_eval_tab()

    def _field_groups(self) -> list[tuple[str, list[tuple[str, str]]]]:
        common = [
            ("随机种子", "RUN.seed"), ("设备", "RUN.device"), ("PyTorch 线程数", "RUN.torch_num_threads"), ("并行环境类型", "RUN.vec_env_type"),
            ("物理步长 dt", "ENV.physical_dt"), ("动作重复", "ENV.action_repeat"),
            ("每回合物理步数", "ENV.max_physical_steps"), ("观测类型", "ENV.observation_type"),
            ("PWM 上限", "ENV.pwm_limit"), ("初始 theta 标准差(度)", "ENV.init_theta_std_deg"),
            ("初始 alpha 标准差(度)", "ENV.init_alpha_std_deg"),
            ("alpha 终止角(度)", "ENV.alpha_abs_limit_deg"),
        ]
        dr = [
            ("启用随机化", "DOMAIN_RANDOMIZATION.enabled"),
            ("初始随机化强度", "DOMAIN_RANDOMIZATION.dr_initial_level"),
            ("最终随机化强度", "DOMAIN_RANDOMIZATION.dr_final_level"),
            ("课程比例", "DOMAIN_RANDOMIZATION.dr_curriculum_fraction"),
            ("摆杆质量比例", "DOMAIN_RANDOMIZATION.param_scale_ranges.m2_scale"),
            ("摆杆质心比例", "DOMAIN_RANDOMIZATION.param_scale_ranges.l2cg_scale"),
            ("电机轴摩擦比例", "DOMAIN_RANDOMIZATION.param_scale_ranges.c_theta_scale"),
            ("摆轴摩擦比例", "DOMAIN_RANDOMIZATION.param_scale_ranges.c_alpha_scale"),
            ("动作延迟步范围", "DOMAIN_RANDOMIZATION.action_delay_steps_range"),
            ("执行器时间常数范围", "DOMAIN_RANDOMIZATION.actuator_tau_range"),
            ("alpha 噪声范围", "DOMAIN_RANDOMIZATION.alpha_sigma_range"),
            ("alpha_dot 噪声范围", "DOMAIN_RANDOMIZATION.alpha_dot_sigma_range"),
        ]
        reward = [
            ("cos(alpha) 权重", "REWARD.k_cos"), ("alpha² 权重", "REWARD.k_alpha"),
            ("alpha_dot² 权重", "REWARD.k_alpha_dot"), ("theta² 权重", "REWARD.k_theta"),
            ("theta_dot² 权重", "REWARD.k_theta_dot"), ("动作惩罚", "REWARD.k_action"),
            ("存活奖励", "REWARD.alive"), ("大角度阈值(度)", "REWARD.angle_penalty_deg"),
        ]
        if self.algorithm == "PPO":
            algo = [
                ("总训练步数", "PPO.total_timesteps"), ("并行环境数", "PPO.n_envs"),
                ("rollout n_steps", "PPO.n_steps"), ("batch size", "PPO.batch_size"),
                ("每批 epochs", "PPO.n_epochs"), ("learning rate", "PPO.learning_rate"),
                ("gamma", "PPO.gamma"), ("GAE lambda", "PPO.gae_lambda"),
                ("clip range", "PPO.clip_range"), ("entropy coef", "PPO.ent_coef"),
                ("Actor 网络", "PPO.net_arch_pi"), ("Critic 网络", "PPO.net_arch_vf"),
                ("激活函数", "PPO.activation_fn_name"),
            ]
        else:
            algo = [
                ("总训练步数", "DQN.total_timesteps"), ("并行环境数", "DQN.n_envs"),
                ("learning rate", "DQN.learning_rate"), ("replay buffer", "DQN.buffer_size"),
                ("learning starts", "DQN.learning_starts"), ("batch size", "DQN.batch_size"),
                ("gamma", "DQN.gamma"), ("train freq", "DQN.train_freq"),
                ("gradient steps", "DQN.gradient_steps"), ("target update", "DQN.target_update_interval"),
                ("初始 epsilon", "DQN.exploration_initial_eps"),
                ("最终 epsilon", "DQN.exploration_final_eps"),
                ("epsilon 衰减常数", "DQN.exploration_decay"),
                ("Double DQN", "DQN.use_double_dqn"), ("Q 网络", "DQN.net_arch"),
                ("激活函数", "DQN.activation_fn_name"),
                ("离散 PWM 动作", "ENV.discrete_actions"),
            ]
        evaluation = [
            ("评估间隔", "EVAL.eval_freq"), ("评估回合数", "EVAL.n_eval_episodes"),
            ("评估最大步数", "EVAL.max_eval_policy_steps"), ("checkpoint 间隔", "EVAL.checkpoint_freq"),
            ("评估随机化强度", "EVAL.eval_randomization_level"),
            ("面板进度刷新步数", "PANEL.progress_update_freq"),
        ]
        return [("环境", common), ("Sim-to-Real 随机化", dr), (self.algorithm, algo), ("奖励", reward), ("评估", evaluation)]

    def _build_basic_tab(self) -> None:
        canvas = tk.Canvas(self.basic_tab, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.basic_tab, orient="vertical", command=canvas.yview)
        self.basic_inner = ttk.Frame(canvas, padding=8)
        self.basic_inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.basic_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row = 0
        for group_name, fields in self._field_groups():
            frame = ttk.LabelFrame(self.basic_inner, text=group_name, padding=8)
            frame.grid(row=row, column=0, sticky="ew", pady=5)
            frame.columnconfigure(1, weight=1)
            row += 1
            for i, (label, path) in enumerate(fields):
                ttk.Label(frame, text=label, width=28).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=2)
                var = tk.StringVar()
                ttk.Entry(frame, textvariable=var, width=55).grid(row=i, column=1, sticky="ew", pady=2)
                ttk.Label(frame, text=path, foreground="#666").grid(row=i, column=2, sticky="w", padx=(8, 0))
                self.entries[path] = var
        self.basic_inner.columnconfigure(0, weight=1)

    def _build_advanced_tab(self) -> None:
        top = ttk.Frame(self.advanced_tab, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="选择任意 config 路径，修改后会直接写入 config.py。").pack(side="left")
        ttk.Button(top, text="应用选中值", command=self.apply_advanced).pack(side="right")

        columns = ("path", "value", "type")
        self.tree = ttk.Treeview(self.advanced_tab, columns=columns, show="headings")
        self.tree.heading("path", text="Config path")
        self.tree.heading("value", text="Value")
        self.tree.heading("type", text="Type")
        self.tree.column("path", width=420, anchor="w")
        self.tree.column("value", width=520, anchor="w")
        self.tree.column("type", width=100, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=8)
        self.tree.bind("<<TreeviewSelect>>", self._advanced_selected)

        edit = ttk.Frame(self.advanced_tab, padding=8)
        edit.pack(fill="x")
        self.advanced_path_var = tk.StringVar()
        self.advanced_edit_var = tk.StringVar()
        ttk.Label(edit, textvariable=self.advanced_path_var, width=55).pack(side="left")
        ttk.Entry(edit, textvariable=self.advanced_edit_var).pack(side="left", fill="x", expand=True, padx=8)

    def _build_progress_tab(self) -> None:
        summary = ttk.LabelFrame(self.progress_tab, text="实时训练状态", padding=8)
        summary.pack(fill="x", padx=8, pady=(8, 4))
        self.run_dir_var = tk.StringVar(value="run_dir: waiting")
        self.train_metric_var = tk.StringVar(value=self.last_train_summary)
        self.eval_metric_var = tk.StringVar(value=self.last_eval_summary)
        ttk.Label(summary, textvariable=self.run_dir_var).pack(anchor="w")
        ttk.Label(summary, textvariable=self.train_metric_var).pack(anchor="w", pady=(3, 0))
        ttk.Label(summary, textvariable=self.eval_metric_var).pack(anchor="w", pady=(3, 0))

        chart_frame = ttk.Frame(self.progress_tab)
        chart_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        if Figure is None or FigureCanvasTkAgg is None:
            self.figure = None
            self.chart_canvas = None
            ttk.Label(
                chart_frame,
                text="Matplotlib Tk backend is unavailable. Training continues; use the Full Log tab for metrics.",
            ).pack(expand=True)
            return
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax_train = self.figure.add_subplot(311)
        self.ax_eval_reward = self.figure.add_subplot(312)
        self.ax_success = self.figure.add_subplot(313)
        self.figure.tight_layout(pad=2.0)
        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_log_tab(self) -> None:
        self.log_text = scrolledtext.ScrolledText(self.log_tab, wrap="word", font=("Menlo", 11))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_eval_tab(self) -> None:
        frame = ttk.Frame(self.eval_tab, padding=16)
        frame.pack(fill="x")
        self.eval_path_var = tk.StringVar(value=str(getattr(config, "EVAL_MODEL_PATH", "")))
        ttk.Label(frame, text="Model .zip:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.eval_path_var, width=90).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="选择", command=self.choose_eval_model).grid(row=0, column=2)
        ttk.Button(frame, text="运行评估", command=self.start_eval).grid(row=1, column=1, sticky="w", pady=12)
        frame.columnconfigure(1, weight=1)

    # ------------------------------------------------------------ config
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
            var.set(repr(value) if not isinstance(value, str) else value)
        self._refresh_advanced()
        self.progress_label.configure(text="config.py reloaded")

    def save_basic(self, quiet: bool = False) -> bool:
        updates: Dict[str, Any] = {}
        try:
            for path, var in self.entries.items():
                if path not in self.current_values:
                    continue
                updates[path] = _parse_text(var.get(), self.current_values[path])
            backup = update_config_file(CONFIG_PATH, updates)
            self.reload_config()
            if not quiet:
                messagebox.showinfo("Saved", f"已写入 config.py\nBackup: {backup.name}")
            return True
        except (ValueError, ConfigEditError, OSError) as exc:
            messagebox.showerror("Save failed", str(exc))
            return False

    def _refresh_advanced(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.advanced_values.clear()
        sections = ["RUN", "ENV", "PHYSICAL_PARAMS", "DOMAIN_RANDOMIZATION", self.algo_key, "REWARD", "EVAL", "SMOKE", "PANEL"]
        for section in sections:
            if not hasattr(config, section):
                continue
            for path, value in _flatten(section, getattr(config, section)):
                self.advanced_values[path] = value
                display = repr(value) if not isinstance(value, str) else value
                self.tree.insert("", "end", iid=path, values=(path, display, type(value).__name__))

    def _advanced_selected(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        path = selected[0]
        value = self.advanced_values[path]
        self.advanced_path_var.set(path)
        self.advanced_edit_var.set(repr(value) if not isinstance(value, str) else value)

    def apply_advanced(self) -> None:
        path = self.advanced_path_var.get()
        if not path:
            return
        try:
            value = _parse_text(self.advanced_edit_var.get(), self.advanced_values[path])
            backup = update_config_file(CONFIG_PATH, {path: value})
            self.reload_config()
            messagebox.showinfo("Saved", f"{path} 已写入 config.py\nBackup: {backup.name}")
        except (ValueError, ConfigEditError, OSError) as exc:
            messagebox.showerror("Save failed", str(exc))

    # ------------------------------------------------------------ process
    def start_process(self, mode: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Running", "已有训练/评估进程正在运行")
            return
        if bool(getattr(config, "PANEL", {}).get("auto_save_before_run", True)) and not self.save_basic(quiet=True):
            return
        self.notebook.select(self.progress_tab)
        self.log_text.delete("1.0", "end")
        self.progress["value"] = 0.0
        self._reset_live_metrics()
        cmd = [sys.executable, "-u", str(PROJECT_ROOT / "run.py"), mode]
        self._launch(cmd)

    def start_eval(self) -> None:
        model_path = self.eval_path_var.get().strip()
        if not model_path:
            messagebox.showerror("Missing model", "请选择模型 .zip")
            return
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Running", "已有训练/评估进程正在运行")
            return
        self.notebook.select(self.progress_tab)
        cmd = [sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "eval", model_path]
        self._launch(cmd)

    def _launch(self, cmd: list[str]) -> None:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            messagebox.showerror("Launch failed", str(exc))
            return
        self.progress_label.configure(text="process started")
        threading.Thread(target=self._reader_thread, daemon=True).start()

    def _reader_thread(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.output_queue.put(line)
        code = self.proc.wait()
        self.output_queue.put(f"\n[PANEL] process exited with code {code}\n")
        self.output_queue.put("__PROCESS_DONE__")

    def _reset_live_metrics(self) -> None:
        self.current_run_dir = ""
        self.train_steps.clear()
        self.train_reward.clear()
        self.train_reward_ma.clear()
        self.train_success_ma.clear()
        for data in self.eval_data.values():
            for values in data.values():
                values.clear()
        self.last_train_summary = "waiting for completed episodes"
        self.last_eval_summary = "waiting for deterministic evaluation"
        if hasattr(self, "run_dir_var"):
            self.run_dir_var.set("run_dir: waiting")
            self.train_metric_var.set(self.last_train_summary)
            self.eval_metric_var.set(self.last_eval_summary)
        self.chart_dirty = True

    @staticmethod
    def _append_limited(values: List[float], value: float, limit: int = 5000) -> None:
        values.append(value)
        if len(values) > limit:
            del values[: len(values) - limit]

    def _handle_metrics_line(self, line: str) -> None:
        run_match = RUN_DIR_RE.search(line.strip())
        if run_match:
            self.current_run_dir = run_match.group("path").strip()
            self.run_dir_var.set(f"run_dir: {self.current_run_dir}")

        match = TRAIN_EPISODE_RE.search(line)
        if match:
            steps = int(match.group("steps"))
            reward = float(match.group("reward"))
            moving = float(match.group("moving"))
            success = float(match.group("success"))
            self._append_limited(self.train_steps, steps)
            self._append_limited(self.train_reward, reward)
            self._append_limited(self.train_reward_ma, moving)
            self._append_limited(self.train_success_ma, success)
            self.last_train_summary = (
                f"Training episode: reward/step={reward:.4f}, moving mean(30)={moving:.4f}, "
                f"completion rate(30)={success * 100.0:.1f}%"
            )
            self.train_metric_var.set(self.last_train_summary)
            self.chart_dirty = True

        eval_match = EVAL_METRICS_RE.search(line)
        if eval_match:
            kind = eval_match.group("kind")
            data = self.eval_data[kind]
            steps = int(eval_match.group("steps"))
            reward = float(eval_match.group("reward"))
            success = float(eval_match.group("success"))
            alpha = float(eval_match.group("alpha"))
            self._append_limited(data["steps"], steps, 1000)
            self._append_limited(data["reward"], reward, 1000)
            self._append_limited(data["success"], success, 1000)
            self._append_limited(data["alpha"], alpha, 1000)
            nom = self.eval_data["nominal"]
            rnd = self.eval_data["randomized"]
            parts = []
            if nom["reward"]:
                parts.append(
                    f"nominal r/step={nom['reward'][-1]:.4f}, success={nom['success'][-1] * 100.0:.1f}%, "
                    f"|alpha|={nom['alpha'][-1]:.4f}"
                )
            if rnd["reward"]:
                parts.append(
                    f"randomized r/step={rnd['reward'][-1]:.4f}, success={rnd['success'][-1] * 100.0:.1f}%, "
                    f"|alpha|={rnd['alpha'][-1]:.4f}"
                )
            self.last_eval_summary = "Deterministic eval: " + " | ".join(parts)
            self.eval_metric_var.set(self.last_eval_summary)
            self.chart_dirty = True

    def _refresh_charts(self) -> None:
        try:
            if self.chart_dirty and self.figure is not None and self.chart_canvas is not None:
                self.ax_train.clear()
                self.ax_eval_reward.clear()
                self.ax_success.clear()

                if self.train_steps:
                    self.ax_train.plot(self.train_steps, self.train_reward, linewidth=0.8, alpha=0.45, label="episode")
                    self.ax_train.plot(self.train_steps, self.train_reward_ma, linewidth=1.8, label="moving mean (30)")
                self.ax_train.set_title("PPO Episode Reward / Step")
                self.ax_train.set_ylabel("reward / step")
                self.ax_train.grid(True, alpha=0.3)
                if self.train_steps:
                    self.ax_train.legend(loc="best")

                for kind, label in (("nominal", "nominal"), ("randomized", "randomized")):
                    data = self.eval_data[kind]
                    if data["steps"]:
                        self.ax_eval_reward.plot(data["steps"], data["reward"], marker="o", markersize=3, label=label)
                        self.ax_success.plot(data["steps"], data["success"], marker="o", markersize=3, label=label)
                self.ax_eval_reward.set_title("Deterministic Evaluation Reward / Step")
                self.ax_eval_reward.set_ylabel("reward / step")
                self.ax_eval_reward.grid(True, alpha=0.3)
                if any(self.eval_data[k]["steps"] for k in self.eval_data):
                    self.ax_eval_reward.legend(loc="best")

                self.ax_success.set_title("Evaluation Success Rate (Completed Full Balance Episode)")
                self.ax_success.set_xlabel("timesteps")
                self.ax_success.set_ylabel("success rate")
                self.ax_success.set_ylim(-0.03, 1.03)
                self.ax_success.grid(True, alpha=0.3)
                if any(self.eval_data[k]["steps"] for k in self.eval_data):
                    self.ax_success.legend(loc="best")

                self.figure.tight_layout(pad=1.5)
                self.chart_canvas.draw_idle()
                self.chart_dirty = False
        finally:
            self.after(1000, self._refresh_charts)

    def _poll_output(self) -> None:
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    self.progress_label.configure(text="process finished")
                    continue
                self.log_text.insert("end", line)
                self.log_text.see("end")
                self._handle_metrics_line(line)
                match = PROGRESS_RE.search(line)
                if match:
                    percent = float(match.group("percent"))
                    self.progress["value"] = percent
                    eta = float(match.group("eta"))
                    self.progress_label.configure(
                        text=f"{match.group('steps')}/{match.group('total')}  {percent:.1f}%  "
                             f"fps={float(match.group('fps')):.0f}  ETA={eta/60:.1f} min"
                    )
                eval_match = EVAL_RE.search(line)
                if eval_match:
                    self.progress_label.configure(
                        text=f"eval@{eval_match.group('steps')}: randomized score={eval_match.group('score')} "
                             f"term={eval_match.group('term')}"
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def stop_process(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.progress_label.configure(text="terminate requested")

    def choose_eval_model(self) -> None:
        selected = filedialog.askopenfilename(initialdir=str(PROJECT_ROOT / "runs"), filetypes=[("SB3 model", "*.zip")])
        if selected:
            self.eval_path_var.set(selected)

    def open_runs(self) -> None:
        runs = (PROJECT_ROOT / str(config.RUN.get("root_log_dir", "./runs"))).resolve()
        runs.mkdir(parents=True, exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(runs)])
        elif os.name == "nt":
            os.startfile(str(runs))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(runs)])

    def _on_close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            if not messagebox.askyesno("Training is running", "关闭面板会终止当前进程，继续吗？"):
                return
            self.proc.terminate()
        self.destroy()


if __name__ == "__main__":
    TrainingPanel().mainloop()
