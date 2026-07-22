"""Full Tk panel for the DQN workflow.

The common tab exposes all operational parameters. The advanced tab is built
dynamically from every config dictionary, so no existing parameter becomes
inaccessible when the training implementation changes.
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
from typing import Any, Dict, Iterable

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
from config_editor import update_config_file

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.py"


def _get_value(dotted: str) -> Any:
    parts = dotted.split(".")
    value: Any = getattr(config, parts[0])
    for part in parts[1:]:
        value = value[part]
    return value


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(f"{prefix}.{key}", child)
    else:
        yield prefix, value


def _parse(text: str, original: Any) -> Any:
    raw = text.strip()
    if isinstance(original, str):
        # Both common and advanced tabs display strings with repr(), e.g.
        # 'auto'. Parse one or more whole-string quote layers so saving does
        # not turn it into the literal value "'auto'". Unquoted text remains
        # valid for convenient editing.
        value = raw
        for _ in range(4):
            if len(value) < 2 or value[0] not in {"'", '"'} or value[-1] != value[0]:
                break
            try:
                parsed = ast.literal_eval(value)
            except Exception:
                break
            if not isinstance(parsed, str) or parsed == value:
                break
            value = parsed.strip()
        return value
    if isinstance(original, bool):
        lowered = raw.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Expected a boolean value, received: {raw}")
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        if isinstance(original, int):
            parsed = int(raw)
        elif isinstance(original, float):
            parsed = float(raw)
        else:
            raise
    if isinstance(original, tuple) and isinstance(parsed, list):
        parsed = tuple(parsed)
    return parsed


def _make_scroll(parent):
    outer = ttk.Frame(parent)
    canvas = tk.Canvas(outer, highlightthickness=0)
    ybar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    xbar = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
    inner = ttk.Frame(canvas, padding=8)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")

    def configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(window, width=max(canvas.winfo_width(), inner.winfo_reqwidth()))

    inner.bind("<Configure>", configure)
    canvas.bind("<Configure>", configure)
    canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    ybar.grid(row=0, column=1, sticky="ns")
    xbar.grid(row=1, column=0, sticky="ew")
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)
    return outer, inner


class TrainingPanel(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DQN")
        self.geometry(str(config.PANEL.get("window_geometry", "1460x920")))
        self.minsize(1180, 760)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.proc: subprocess.Popen | None = None
        self.output_queue: queue.Queue = queue.Queue()
        self.active_stage = "idle"
        self.last_run_dir = ""
        self.basic_entries: Dict[str, tk.StringVar] = {}
        self.advanced_entries: Dict[str, tk.StringVar] = {}
        self.lockable_widgets: list[tk.Widget] = []
        self.pending_event: Dict[str, Any] | None = None

        self.episode_steps: list[int] = []
        self.episode_rewards: list[float] = []
        self.eval_steps: list[int] = []
        self.eval_rewards: list[float] = []
        self.eval_success: list[float] = []
        self._plot_pending = False

        self.status_var = tk.StringVar(value="Ready")
        self.stage_var = tk.StringVar(value="Stage: not started")
        self.progress_var = tk.StringVar(value="First 2,000,000 steps: nominal; staged randomization afterward")
        self.repro_var = tk.StringVar(value="6→64→64→10 · vanilla DQN · LR=5e-4 · behavior snapshot=50")

        self._build()
        self.reload_config()
        self.after(100, self._poll_output)

    def _field_groups(self):
        return [
            ("Environment and Observation", [
                ("Physics time step", "ENV.physical_dt"),
                ("Maximum episode steps", "ENV.max_physical_steps"),
                ("Theta position limit", "ENV.theta_limit"),
                ("Theta-dot limit", "ENV.theta_dot_limit"),
                ("Alpha-dot limit", "ENV.alpha_dot_limit"),
                ("PWM limit", "ENV.pwm_limit"),
                ("10-level action table", "ENV.discrete_actions"),
                ("Initial theta mean / deg", "ENV.init_theta_mean_deg"),
                ("Initial theta std / deg", "ENV.init_theta_std_deg"),
                ("Initial alpha mean / deg", "ENV.init_alpha_mean_deg"),
                ("Initial alpha std / deg", "ENV.init_alpha_std_deg"),
                ("Velocity LPF coefficient", "ENV.velocity_lpf"),
                ("Clip velocity inputs", "ENV.clip_velocity_in_observation"),
            ]),
            ("DQN Core", [
                ("Total training steps", "DQN.total_timesteps"),
                ("Learning rate", "DQN.learning_rate"),
                ("Adam beta1", "DQN.adam_beta1"),
                ("Adam beta2", "DQN.adam_beta2"),
                ("Adam eps", "DQN.adam_eps"),
                ("Replay buffer", "DQN.buffer_size"),
                ("Learning starts", "DQN.learning_starts"),
                ("Batch size", "DQN.batch_size"),
                ("Gamma", "DQN.gamma"),
                ("Train freq", "DQN.train_freq"),
                ("Gradient steps", "DQN.gradient_steps"),
                ("Target sync interval", "DQN.target_update_interval"),
                ("Behavior snapshot interval", "DQN.behavior_snapshot_interval"),
                ("Weight initialization std", "DQN.weight_init_std"),
                ("Huber delta", "DQN.huber_delta"),
                ("Double DQN (False for MATLAB reproduction)", "DQN.use_double_dqn"),
                ("Initial epsilon", "DQN.exploration_initial_eps"),
                ("Final epsilon", "DQN.exploration_final_eps"),
                ("Epsilon decay constant", "DQN.exploration_decay"),
                ("Network architecture", "DQN.net_arch"),
                ("Save interval", "DQN.save_every_steps"),
                ("Stability averaging window", "DQN.stop_avg_window"),
                ("Stable-step threshold", "DQN.stop_avg_stable_steps_threshold"),
                ("Allow early randomization after nominal success", "DQN.allow_early_nominal_stage_transition"),
                ("Reset exploration in randomized stages", "DQN.reset_exploration_each_randomized_stage"),
                ("Randomized-stage initial epsilon", "DQN.randomized_stage_initial_eps"),
                ("Randomized-stage final epsilon", "DQN.randomized_stage_final_eps"),
                ("Randomized-stage epsilon decay", "DQN.randomized_stage_exploration_decay"),
            ]),
            ("Fixed Staged Randomization After 2M Steps", [
                ("Enable randomization", "DOMAIN_RANDOMIZATION.enabled"),
                ("Stage names", "DOMAIN_RANDOMIZATION.training_stage_names"),
                ("Stage levels", "DOMAIN_RANDOMIZATION.training_stage_levels"),
                ("Explicit stage steps", "DOMAIN_RANDOMIZATION.training_stage_steps"),
                ("Stage fraction description", "DOMAIN_RANDOMIZATION.training_stage_fractions"),
                ("Nominal steps", "DOMAIN_RANDOMIZATION.nominal_recovery_steps"),
                ("Clear replay at stage transitions", "DOMAIN_RANDOMIZATION.clear_replay_between_stages"),
                ("Reset Adam at stage transitions", "DOMAIN_RANDOMIZATION.reset_optimizer_between_stages"),
                ("Sync target at stage transitions", "DOMAIN_RANDOMIZATION.sync_target_between_stages"),
                ("Sync behavior network at stage transitions", "DOMAIN_RANDOMIZATION.sync_behavior_snapshot_between_stages"),
                ("New-stage warm-up", "DOMAIN_RANDOMIZATION.stage_replay_warmup_steps"),
                ("Initial theta std range", "DOMAIN_RANDOMIZATION.init_theta_std_deg_range"),
                ("Initial alpha std range", "DOMAIN_RANDOMIZATION.init_alpha_std_deg_range"),
                ("Initial theta-dot std range", "DOMAIN_RANDOMIZATION.init_theta_dot_std_range"),
                ("Initial alpha-dot std range", "DOMAIN_RANDOMIZATION.init_alpha_dot_std_range"),
                ("PWM-limit scale range", "DOMAIN_RANDOMIZATION.pwm_limit_scale_range"),
                ("PWM gain range", "DOMAIN_RANDOMIZATION.pwm_gain_range"),
                ("PWM bias range", "DOMAIN_RANDOMIZATION.pwm_bias_range"),
                ("PWM dead-zone range", "DOMAIN_RANDOMIZATION.pwm_deadzone_range"),
                ("PWM noise range", "DOMAIN_RANDOMIZATION.pwm_noise_sigma_range"),
                ("Actuator tau range", "DOMAIN_RANDOMIZATION.actuator_tau_range"),
                ("Action-delay steps", "DOMAIN_RANDOMIZATION.action_delay_steps_range"),
                ("theta bias", "DOMAIN_RANDOMIZATION.theta_bias_range"),
                ("alpha bias", "DOMAIN_RANDOMIZATION.alpha_bias_range"),
                ("Theta noise range", "DOMAIN_RANDOMIZATION.theta_sigma_range"),
                ("Alpha noise range", "DOMAIN_RANDOMIZATION.alpha_sigma_range"),
                ("Theta-dot noise range", "DOMAIN_RANDOMIZATION.theta_dot_sigma_range"),
                ("Alpha-dot noise range", "DOMAIN_RANDOMIZATION.alpha_dot_sigma_range"),
                ("Encoder quantization range", "DOMAIN_RANDOMIZATION.encoder_quantization_rad_range"),
                ("Random LPF probability", "DOMAIN_RANDOMIZATION.use_lpf_velocity_probability"),
                ("LPF range", "DOMAIN_RANDOMIZATION.velocity_lpf_range"),
            ]),
            ("Physical Parameter Randomization Ranges", [
                ("g", "DOMAIN_RANDOMIZATION.param_scale_ranges.g_scale"),
                ("m1", "DOMAIN_RANDOMIZATION.param_scale_ranges.m1_scale"),
                ("m2", "DOMAIN_RANDOMIZATION.param_scale_ranges.m2_scale"),
                ("l1", "DOMAIN_RANDOMIZATION.param_scale_ranges.l1_scale"),
                ("l1cg", "DOMAIN_RANDOMIZATION.param_scale_ranges.l1cg_scale"),
                ("l2cg", "DOMAIN_RANDOMIZATION.param_scale_ranges.l2cg_scale"),
                ("I1z", "DOMAIN_RANDOMIZATION.param_scale_ranges.I1z_scale"),
                ("I2x", "DOMAIN_RANDOMIZATION.param_scale_ranges.I2x_scale"),
                ("I2y", "DOMAIN_RANDOMIZATION.param_scale_ranges.I2y_scale"),
                ("I2z", "DOMAIN_RANDOMIZATION.param_scale_ranges.I2z_scale"),
                ("c_theta", "DOMAIN_RANDOMIZATION.param_scale_ranges.c_theta_scale"),
                ("c_alpha", "DOMAIN_RANDOMIZATION.param_scale_ranges.c_alpha_scale"),
                ("k_t", "DOMAIN_RANDOMIZATION.param_scale_ranges.k_t_scale"),
                ("k_b", "DOMAIN_RANDOMIZATION.param_scale_ranges.k_b_scale"),
                ("k_u", "DOMAIN_RANDOMIZATION.param_scale_ranges.k_u_scale"),
                ("R", "DOMAIN_RANDOMIZATION.param_scale_ranges.R_scale"),
            ]),
            ("Reward", [
                ("cos(alpha)", "REWARD.k_cos_alpha"),
                ("alpha_dot²", "REWARD.k_alpha_dot"),
                ("theta_dot²", "REWARD.k_theta_dot"),
                ("theta²", "REWARD.k_theta"),
                ("Angle-penalty threshold", "REWARD.alpha_penalty_deg"),
                ("Angle-penalty value", "REWARD.alpha_penalty_value"),
                ("Action L2 penalty", "REWARD.action_l2"),
            ]),
            ("Evaluation, Best Model, and Early Stopping", [
                ("Evaluation frequency", "EVAL.eval_freq"),
                ("Evaluation episodes", "EVAL.n_eval_episodes"),
                ("Maximum evaluation steps", "EVAL.max_eval_policy_steps"),
                ("Checkpoint frequency", "EVAL.checkpoint_freq"),
                ("Nominal evaluation level", "EVAL.nominal_eval_randomization_level"),
                ("Randomized evaluation level", "EVAL.randomized_eval_randomization_level"),
                ("Capture angle", "EVAL.capture_angle_deg"),
                ("Stable alpha-dot limit", "EVAL.stable_alpha_dot_max"),
                ("Stable hold steps", "EVAL.stable_hold_steps"),
                ("Early stopping in randomized stages", "EVAL.early_stop_enabled"),
                ("Early-stop start fraction", "EVAL.early_stop_start_fraction"),
                ("Early-stop patience evaluations", "EVAL.early_stop_patience_evals"),
                ("Minimum success rate", "EVAL.early_stop_min_success_rate"),
                ("Randomized-model selection threshold", "EVAL.randomized_model_min_success_for_selection"),
            ]),
            ("Test", [
                ("Test duration", "TEST.duration_seconds"),
                ("Test randomization level", "TEST.randomization_level"),
                ("Test seed", "TEST.seed"),
            ]),
            ("Runtime", [
                ("Log directory", "RUN.root_log_dir"),
                ("Random seed", "RUN.seed"),
                ("Device", "RUN.device"),
                ("Torch threads", "RUN.torch_num_threads"),
                ("Deterministic algorithms", "RUN.torch_deterministic"),
            ]),
        ]

    def _build(self) -> None:
        top = ttk.Frame(self, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, text="Single entry point: python run.py", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Label(top, text="First 2M steps nominal; then levels 0.10 → 0.30 → 0.50", foreground="#444").pack(side="left", padx=18)
        ttk.Button(top, text="Open Runs Directory", command=self.open_runs).pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.basic_tab = ttk.Frame(self.notebook)
        self.advanced_tab = ttk.Frame(self.notebook)
        self.workflow_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.basic_tab, text="Common Parameters")
        self.notebook.add(self.advanced_tab, text="All Config Parameters")
        self.notebook.add(self.workflow_tab, text="Training, Testing, and Curves")
        self.notebook.add(self.log_tab, text="Full Log")

        self._build_basic()
        self._build_advanced()
        self._build_workflow()
        self._build_log()

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bottom, text="Close Panel", command=self._on_close).pack(side="right")

    def _build_basic(self) -> None:
        outer, inner = _make_scroll(self.basic_tab)
        outer.pack(fill="both", expand=True)
        row = 0
        for group, fields in self._field_groups():
            box = ttk.LabelFrame(inner, text=group, padding=8)
            box.grid(row=row, column=0, sticky="ew", pady=5)
            box.columnconfigure(1, weight=1)
            for i, (label, path) in enumerate(fields):
                ttk.Label(box, text=label, width=34).grid(row=i, column=0, sticky="w", pady=2)
                var = tk.StringVar()
                entry = ttk.Entry(box, textvariable=var, width=85)
                entry.grid(row=i, column=1, sticky="ew", pady=2)
                ttk.Label(box, text=path, foreground="#666").grid(row=i, column=2, sticky="w", padx=8)
                self.basic_entries[path] = var
                self.lockable_widgets.append(entry)
            row += 1
        buttons = ttk.Frame(inner, padding=8)
        buttons.grid(row=row, column=0, sticky="ew")
        save = ttk.Button(buttons, text="Save Common Parameters", command=lambda: self.save_entries(self.basic_entries))
        save.pack(side="left")
        reload_button = ttk.Button(buttons, text="Reload", command=self.reload_config)
        reload_button.pack(side="left", padx=8)
        self.lockable_widgets.extend([save, reload_button])

    def _build_advanced(self) -> None:
        self.advanced_outer, self.advanced_inner = _make_scroll(self.advanced_tab)
        self.advanced_outer.pack(fill="both", expand=True)

    def _rebuild_advanced(self) -> None:
        for child in self.advanced_inner.winfo_children():
            child.destroy()
        self.advanced_entries.clear()
        row = 0
        for section in config.config_sections():
            value = getattr(config, section)
            box = ttk.LabelFrame(self.advanced_inner, text=section, padding=8)
            box.grid(row=row, column=0, sticky="ew", pady=5)
            box.columnconfigure(1, weight=1)
            for i, (path, current) in enumerate(_flatten(section, value)):
                ttk.Label(box, text=path, width=64).grid(row=i, column=0, sticky="w", pady=1)
                var = tk.StringVar(value=repr(current))
                entry = ttk.Entry(box, textvariable=var, width=90)
                entry.grid(row=i, column=1, sticky="ew", pady=1)
                self.advanced_entries[path] = var
                self.lockable_widgets.append(entry)
            row += 1
        buttons = ttk.Frame(self.advanced_inner, padding=8)
        buttons.grid(row=row, column=0, sticky="ew")
        save = ttk.Button(buttons, text="Save All Parameter Changes", command=lambda: self.save_entries(self.advanced_entries))
        save.pack(side="left")
        ttk.Button(buttons, text="Reload", command=self.reload_config).pack(side="left", padx=8)

    def _build_workflow(self) -> None:
        info = ttk.LabelFrame(self.workflow_tab, text="Workflow and Current Status", padding=10)
        info.pack(fill="x", padx=8, pady=6)
        ttk.Label(info, textvariable=self.repro_var, font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, columnspan=6, sticky="w")
        self.progress = ttk.Progressbar(info, maximum=100)
        self.progress.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(8, 4))
        ttk.Label(info, textvariable=self.progress_var).grid(row=2, column=0, columnspan=6, sticky="w")
        ttk.Label(info, textvariable=self.stage_var).grid(row=3, column=0, columnspan=6, sticky="w", pady=3)
        info.columnconfigure(5, weight=1)

        controls = ttk.Frame(info)
        controls.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        self.train_button = ttk.Button(controls, text="Start Full Training", command=self.start_training)
        self.train_button.pack(side="left")
        self.smoke_button = ttk.Button(controls, text="Run Full Smoke Test", command=self.start_smoke)
        self.smoke_button.pack(side="left", padx=6)
        self.test_button = ttk.Button(controls, text="Start 30-Second Test", command=self.start_test, state="disabled")
        self.test_button.pack(side="left", padx=6)
        ttk.Button(controls, text="Select Existing run_dir", command=self.choose_run_dir).pack(side="left", padx=6)
        self.stop_button = ttk.Button(controls, text="Emergency Stop Current Process", command=self.stop_process, state="disabled")
        self.stop_button.pack(side="right")

        self.figure = Figure(figsize=(11, 7), dpi=100)
        self.ax_reward = self.figure.add_subplot(311)
        self.ax_eval = self.figure.add_subplot(312)
        self.ax_success = self.figure.add_subplot(313)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.workflow_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=5)

    def _build_log(self) -> None:
        self.log_text = scrolledtext.ScrolledText(self.log_tab, wrap="word", font=("Menlo", 10))
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def reload_config(self) -> None:
        importlib.invalidate_caches()
        importlib.reload(config)
        for path, var in self.basic_entries.items():
            try:
                var.set(repr(_get_value(path)))
            except Exception as exc:
                var.set(f"<ERROR: {exc}>")
        self._rebuild_advanced()
        self.repro_var.set(
            f"6-D input · Network 6→{config.DQN['net_arch'][0]}→{config.DQN['net_arch'][1]}→{len(config.ENV['discrete_actions'])} · "
            f"vanilla={not config.DQN['use_double_dqn']} · LR={config.DQN['learning_rate']} · behavior={config.DQN['behavior_snapshot_interval']}"
        )
        self.status_var.set("Configuration reloaded")

    def save_entries(self, entries: Dict[str, tk.StringVar]) -> None:
        try:
            updates = {path: _parse(var.get(), _get_value(path)) for path, var in entries.items()}
            backup = update_config_file(CONFIG_PATH, updates)
            self.reload_config()
            messagebox.showinfo("Save Successful", f"config.py updated\nBackup: {backup.name}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def _autosave_basic(self) -> bool:
        if not bool(config.PANEL.get("auto_save_before_run", True)):
            return True
        try:
            updates = {path: _parse(var.get(), _get_value(path)) for path, var in self.basic_entries.items()}
            update_config_file(CONFIG_PATH, updates)
            importlib.reload(config)
            return True
        except Exception as exc:
            messagebox.showerror("Configuration Save Failed", str(exc))
            return False

    def _reset_curves(self) -> None:
        self.episode_steps.clear(); self.episode_rewards.clear()
        self.eval_steps.clear(); self.eval_rewards.clear(); self.eval_success.clear()
        self.progress["value"] = 0
        self._redraw_plot()

    def start_training(self) -> None:
        if not self._autosave_basic():
            return
        self._reset_curves()
        self._launch([sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "--worker", "train"], "training")
        self.notebook.select(self.workflow_tab)

    def start_smoke(self) -> None:
        if not self._autosave_basic():
            return
        self._reset_curves()
        self._launch([sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "--worker", "smoke"], "smoke")
        self.notebook.select(self.workflow_tab)

    def choose_run_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Select Training run_dir", initialdir=str(PROJECT_ROOT / config.RUN["root_log_dir"]))
        if chosen:
            self.last_run_dir = chosen
            self.test_button.configure(state="normal")
            self.status_var.set(f"Selected: {chosen}")

    def start_test(self) -> None:
        if not self.last_run_dir:
            self.choose_run_dir()
        if self.last_run_dir:
            self._launch([
                sys.executable, "-u", str(PROJECT_ROOT / "run.py"), "--worker", "test",
                "--run-dir", self.last_run_dir, "--variant", "current",
            ], "test")

    def _set_locked(self, locked: bool) -> None:
        state = "disabled" if locked else "normal"
        for widget in self.lockable_widgets:
            try:
                widget.configure(state=state)
            except Exception:
                pass
        self.train_button.configure(state=state)
        self.smoke_button.configure(state=state)

    def _launch(self, cmd: list[str], stage: str) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Task Already Running", "Wait for or stop the current process first.")
            return
        self.active_stage = stage
        self.pending_event = None
        self.stop_button.configure(state="normal")
        self._set_locked(bool(config.PANEL.get("lock_config_while_running", True)))
        self.status_var.set(f"{stage} running")
        self.log_text.insert("end", "\n$ " + " ".join(cmd) + "\n")
        self.log_text.see("end")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except OSError as exc:
            self._set_locked(False)
            messagebox.showerror("Launch Failed", str(exc))
            return
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            self.output_queue.put(("line", line))
        self.output_queue.put(("done", self.proc.wait()))

    def _poll_output(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "line":
                    self._handle_line(str(payload))
                else:
                    self._handle_done(int(payload))
        except queue.Empty:
            pass
        self.after(100, self._poll_output)

    def _handle_line(self, line: str) -> None:
        self.log_text.insert("end", line)
        self.log_text.see("end")
        marker = "[PANEL_JSON] "
        if marker not in line:
            return
        try:
            event = json.loads(line.split(marker, 1)[1])
        except Exception:
            return
        if event.get("run_dir"):
            self.last_run_dir = str(event["run_dir"])
        typ = event.get("event")
        if typ == "training_started":
            self.status_var.set(f"Training started: {event.get('run_dir')}")
        elif typ == "training_stage":
            self.stage_var.set(
                f"Stage {event.get('stage_index')}/{event.get('stage_count')} · {event.get('stage_name')} · "
                f"level={float(event.get('level', 0)):.2f} · steps={int(event.get('stage_steps', 0)):,} · "
                f"replay cleared={event.get('replay_cleared', False)}"
            )
        elif typ == "stage_transition":
            self.stage_var.set(
                f"Transitioned to {event.get('stage_name')} level={float(event.get('level', 0)):.2f} · "
                f"replay {event.get('replay_size_before')}→{event.get('replay_size_after')} · "
                f"Adam reset={event.get('optimizer_reset')} · target synced={event.get('target_synced')} · "
                f"behavior synced={event.get('behavior_synced')}"
            )
        elif typ == "progress":
            step = int(event.get("step", 0)); total = max(1, int(event.get("total", config.DQN["total_timesteps"])))
            self.progress["value"] = min(100.0, 100.0 * step / total)
            self.progress_var.set(
                f"step={step:,}/{total:,} · level={float(event.get('level', 0)):.2f} · "
                f"replay={event.get('replay_size')} · epsilon={float(event.get('epsilon', 0)):.5f}"
            )
        elif typ == "episode":
            self.episode_steps.append(int(event.get("step", 0)))
            self.episode_rewards.append(float(event.get("reward", 0.0)))
            limit = int(config.PANEL.get("episode_curve_max_points", 2000))
            self.episode_steps = self.episode_steps[-limit:]
            self.episode_rewards = self.episode_rewards[-limit:]
            self._schedule_plot()
        elif typ == "evaluation":
            self.eval_steps.append(int(event.get("step", 0)))
            self.eval_rewards.append(float(event.get("mean_reward", 0.0)))
            self.eval_success.append(float(event.get("stable_success_rate", 0.0)))
            self.progress_var.set(
                f"{event.get('phase')} eval level={float(event.get('eval_randomization_level', 0)):.2f} @"
                f"{int(event.get('step', 0)):,}: reward/step={float(event.get('mean_reward', 0)):.5f}, "
                f"success={100*float(event.get('stable_success_rate', 0)):.1f}%"
            )
            self._schedule_plot()
        elif typ == "nominal_success":
            self.stage_var.set(
                f"Stability criterion reached @ {int(event.get('step', 0)):,} · "
                f"last-window maintain={float(event.get('recent_mean_maintain', 0)):.1f} · snapshot saved"
            )
        elif typ == "early_stop":
            self.stage_var.set("Randomized-stage early stop: " + str(event.get("reason", "")))
        elif typ in {"training_finished", "test_finished"}:
            self.pending_event = event

    def _handle_done(self, code: int) -> None:
        stage = self.active_stage
        self.proc = None
        self.active_stage = "idle"
        self.stop_button.configure(state="disabled")
        self._set_locked(False)
        if code != 0:
            self.status_var.set(f"{stage} failed, exit code {code}")
            messagebox.showerror("Process Failed", f"{stage} exited with code {code}. See the full log.")
            return
        if self.last_run_dir:
            self.test_button.configure(state="normal")
        self.status_var.set(f"{stage} completed")
        if stage == "smoke":
            messagebox.showinfo("Smoke Test Completed", "Training, stage transition, deployment-model export, and the 0.25-second test all completed.")
        elif stage == "test" and self.pending_event:
            result = self.pending_event.get("result", {})
            messagebox.showinfo("Test Completed", f"Result: {result.get('result_json', self.last_run_dir)}")

    def _schedule_plot(self) -> None:
        if not self._plot_pending:
            self._plot_pending = True
            self.after(250, self._redraw_plot)

    def _redraw_plot(self) -> None:
        self._plot_pending = False
        self.ax_reward.clear(); self.ax_eval.clear(); self.ax_success.clear()
        self.ax_reward.set_title("DQN episode reward / step")
        self.ax_reward.set_xlabel("timesteps"); self.ax_reward.grid(True)
        if self.episode_rewards:
            self.ax_reward.plot(self.episode_steps, self.episode_rewards, linewidth=0.8, alpha=0.5, label="episode")
            if len(self.episode_rewards) >= 5:
                window = min(30, len(self.episode_rewards))
                smooth = np.convolve(np.asarray(self.episode_rewards), np.ones(window) / window, mode="valid")
                self.ax_reward.plot(self.episode_steps[window - 1:], smooth, linewidth=1.6, label=f"moving mean ({window})")
            self.ax_reward.legend(loc="best")
        self.ax_eval.set_title("Fixed-Level Evaluation Reward / Step")
        self.ax_eval.set_xlabel("timesteps"); self.ax_eval.grid(True)
        if self.eval_rewards:
            self.ax_eval.plot(self.eval_steps, self.eval_rewards, marker="o")
        self.ax_success.set_title("Stable Success Rate (Hold-Step Threshold)")
        self.ax_success.set_xlabel("timesteps"); self.ax_success.set_ylim(-0.02, 1.02); self.ax_success.grid(True)
        if self.eval_success:
            self.ax_success.plot(self.eval_steps, self.eval_success, marker="s")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def stop_process(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if messagebox.askyesno("Emergency Stop", "Terminate the current process?"):
            self.proc.terminate()
            self.status_var.set("Stop requested")

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
            messagebox.showwarning("Cannot Close", "A training or test process is still running.")
            return
        self.destroy()


def main() -> None:
    TrainingPanel().mainloop()


if __name__ == "__main__":
    main()
