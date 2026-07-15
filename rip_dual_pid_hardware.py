from __future__ import annotations

import csv
import json
import math
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import serial
from serial.tools import list_ports

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

BAUD = 921600
CONTROL_DT = 0.005
ARM_LENGTH = 0.18
PEND_LENGTH = 0.24
MOTOR_RADIUS = 0.035
MOTOR_HEIGHT = 0.08
ARM_Z = MOTOR_HEIGHT
MODE_NAMES = {
    0: "DISABLED",
    1: "SWING_PUMP",
    2: "BLEND",
    3: "PID",
}


@dataclass
class PanelConfig:
    duration: float = 10.0
    kp_alpha: float = 0.0
    ki_alpha: float = 0.0
    kd_alpha: float = 0.0
    kp_theta: float = 0.0
    ki_theta: float = 0.0
    kd_theta: float = 0.0
    pwm_limit: float = 150.0
    swing_pwm: float = 120.0
    kick_time: float = 0.10
    enter_deg: float = 15.0
    exit_deg: float = 25.0
    blend_alpha: float = 0.18
    alpha_i_limit: float = 0.50
    theta_i_limit: float = 1.00
    velocity_lpf: float = 0.25
    pot_up: int = 0
    pot_down: int = 0
    theta_rad_per_count: float = 0.00583730846
    motor_sign: int = 1


@dataclass
class HardwareResult:
    time: np.ndarray
    theta: np.ndarray
    theta_dot: np.ndarray
    alpha: np.ndarray
    alpha_dot: np.ndarray
    pwm: np.ndarray
    mode: np.ndarray
    blend: np.ndarray


def stable_phase_start_index(
    result: HardwareResult,
    threshold_deg: float = 15.0,
) -> Optional[int]:
    if result.alpha.size == 0:
        return None
    inside = np.abs(result.alpha) <= math.radians(float(threshold_deg))
    outside = np.flatnonzero(~inside)
    start = int(outside[-1] + 1) if outside.size else 0
    if start >= inside.size:
        return None
    return start


def result_metrics(result: HardwareResult) -> Dict[str, float]:
    stable_index = stable_phase_start_index(result, 15.0)
    max_abs_theta = (
        float(np.max(np.abs(result.theta)))
        if result.theta.size
        else math.nan
    )
    if stable_index is None:
        return {
            "stable_start_time": math.nan,
            "stable_duration": 0.0,
            "alpha_abs_mean": math.nan,
            "alpha_abs_std": math.nan,
            "pwm_abs_mean": math.nan,
            "pwm_abs_std": math.nan,
            "max_abs_theta": max_abs_theta,
        }
    stable_alpha_abs = np.abs(result.alpha[stable_index:])
    stable_pwm_abs = np.abs(result.pwm[stable_index:])
    return {
        "stable_start_time": float(result.time[stable_index]),
        "stable_duration": float(
            result.time[-1] - result.time[stable_index]
        ),
        "alpha_abs_mean": float(np.mean(stable_alpha_abs)),
        "alpha_abs_std": float(np.std(stable_alpha_abs)),
        "pwm_abs_mean": float(np.mean(stable_pwm_abs)),
        "pwm_abs_std": float(np.std(stable_pwm_abs)),
        "max_abs_theta": max_abs_theta,
    }


def build_result_figure(
    result: HardwareResult,
    title_suffix: str = "Hardware Dual PID",
) -> Figure:
    metrics = result_metrics(result)
    fig = Figure(figsize=(10.5, 7.6), tight_layout=True)
    fig.patch.set_facecolor("white")
    axs = fig.subplots(2, 2)
    ax_alpha, ax_theta = axs[0, 0], axs[0, 1]
    ax_pwm, ax_hist = axs[1, 0], axs[1, 1]
    t = result.time
    alpha = result.alpha
    theta = result.theta
    pwm = result.pwm
    fig.suptitle(
        f"Rotary Inverted Pendulum {title_suffix} Response",
        fontsize=15,
        fontweight="bold",
    )
    ax_alpha.plot(t, alpha, linewidth=1.8, label=r"$\alpha$")
    ax_alpha.axhline(0.0, linewidth=1.0, linestyle="--")
    ax_alpha.axhline(math.radians(15.0), linewidth=0.8, linestyle=":")
    ax_alpha.axhline(-math.radians(15.0), linewidth=0.8, linestyle=":")
    ax_alpha.set_title(r"Pendulum Angle $\alpha(t)$", fontsize=12, fontweight="bold")
    ax_alpha.set_xlabel("Time / s")
    ax_alpha.set_ylabel(r"$\alpha$ / rad")
    ax_alpha.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_alpha.legend(loc="lower right", frameon=True)
    stable_index = stable_phase_start_index(result, 15.0)
    if stable_index is None:
        alpha_text = (
            "Stable phase: not reached\n"
            r"criterion: $|\alpha|\leq15^\circ$ until the end"
        )
    else:
        alpha_text = (
            r"$\mathrm{mean}(|\alpha|)$"
            + f" = {metrics['alpha_abs_mean']:.6f} rad\n"
            + r"$\mathrm{std}(|\alpha|)$"
            + f" = {metrics['alpha_abs_std']:.6f} rad\n"
            + f"stable from t = {metrics['stable_start_time']:.3f} s"
        )
        ax_alpha.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_alpha.text(
        0.98,
        0.96,
        alpha_text,
        transform=ax_alpha.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.85,
            edgecolor="0.35",
        ),
    )
    ax_theta.plot(t, theta, linewidth=1.8, label=r"$\theta$")
    ax_theta.axhline(0.0, linewidth=1.0, linestyle="--")
    ax_theta.set_title(r"Rotary Arm Angle $\theta(t)$", fontsize=12, fontweight="bold")
    ax_theta.set_xlabel("Time / s")
    ax_theta.set_ylabel(r"$\theta$ / rad")
    ax_theta.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_theta.legend(loc="upper right", frameon=True)
    ax_pwm.plot(t, pwm, linewidth=1.5, label="PWM")
    ax_pwm.axhline(0.0, linewidth=1.0, linestyle="--")
    ax_pwm.set_title("Control Input PWM(t)", fontsize=12, fontweight="bold")
    ax_pwm.set_xlabel("Time / s")
    ax_pwm.set_ylabel("PWM")
    limit = max(160.0, float(np.max(np.abs(pwm))) * 1.10)
    ax_pwm.set_ylim(-limit, limit)
    ax_pwm.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax_pwm.legend(loc="lower right", frameon=True)
    if stable_index is None:
        pwm_text = "Stable phase: not reached"
    else:
        pwm_text = (
            r"$\mathrm{mean}(|PWM|)$"
            + f" = {metrics['pwm_abs_mean']:.3f}\n"
            + r"$\mathrm{std}(|PWM|)$"
            + f" = {metrics['pwm_abs_std']:.3f}"
        )
        ax_pwm.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_pwm.text(
        0.98,
        0.96,
        pwm_text,
        transform=ax_pwm.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.85,
            edgecolor="0.35",
        ),
    )
    bins = np.arange(-255, 271, 15)
    if stable_index is None:
        ax_hist.text(
            0.5,
            0.5,
            "No final stable phase",
            transform=ax_hist.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
    else:
        ax_hist.hist(
            pwm[stable_index:],
            bins=bins,
            edgecolor="black",
            linewidth=0.45,
        )
    ax_hist.axvline(0.0, linewidth=1.0, linestyle="--")
    ax_hist.set_xlim(-255, 255)
    ax_hist.set_title("Stable-stage PWM Distribution", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("PWM")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    xmax = max(float(t[-1]), 0.1)
    for ax in (ax_alpha, ax_theta, ax_pwm):
        ax.set_xlim(0.0, xmax)
    for ax in (ax_alpha, ax_theta, ax_pwm, ax_hist):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", length=4, width=0.8)
    return fig


def save_result_csv(result: HardwareResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "theta_rad",
                "theta_dot_rad_s",
                "alpha_rad",
                "alpha_dot_rad_s",
                "pwm",
                "mode",
                "blend",
            ]
        )
        for row in zip(
            result.time,
            result.theta,
            result.theta_dot,
            result.alpha,
            result.alpha_dot,
            result.pwm,
            result.mode,
            result.blend,
        ):
            writer.writerow(row)


def default_output_paths(output_dir: str, duration_s: float) -> Tuple[str, str]:
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    duration_tag = f"{float(duration_s):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return (
        os.path.join(output_dir, f"rip_pid_hardware_{duration_tag}s_{stamp}.png"),
        os.path.join(output_dir, f"rip_pid_hardware_{duration_tag}s_{stamp}.csv"),
    )


def rip_points(theta: float, alpha: float):
    center = np.array([0.0, 0.0, ARM_Z])
    radial = np.array([math.cos(theta), math.sin(theta), 0.0])
    tangent = np.array([-math.sin(theta), math.cos(theta), 0.0])
    vertical = np.array([0.0, 0.0, 1.0])
    joint = center + ARM_LENGTH * radial
    pendulum_direction = math.sin(alpha) * tangent + math.cos(alpha) * vertical
    tip = joint + PEND_LENGTH * pendulum_direction
    reference_tip = joint + PEND_LENGTH * vertical
    return center, joint, tip, reference_tip, tangent


def set_line3d(line, p0, p1) -> None:
    line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    line.set_3d_properties([p0[2], p1[2]])


def set_point3d(point, p) -> None:
    point.set_data([p[0]], [p[1]])
    point.set_3d_properties([p[2]])


class SerialReadThread(threading.Thread):
    def __init__(
        self,
        port: serial.Serial,
        line_queue: queue.Queue,
        error_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.line_queue = line_queue
        self.error_queue = error_queue
        self.stop_event = threading.Event()

    def run(self) -> None:
        buffer = bytearray()
        while not self.stop_event.is_set():
            try:
                count = self.port.in_waiting
                data = self.port.read(count if count > 0 else 1)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.error_queue.put(str(exc))
                break
            if not data:
                continue
            buffer.extend(data)
            if len(buffer) > 262144:
                buffer.clear()
            while b"\n" in buffer:
                index = buffer.index(b"\n")
                raw = buffer[:index]
                del buffer[: index + 1]
                line = raw.decode(errors="ignore").strip()
                if line:
                    self.line_queue.put(line)

    def stop(self) -> None:
        self.stop_event.set()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RIP Hardware Dual PID Control Panel - Student")
        self.resize(1240, 820)
        self.serial: Optional[serial.Serial] = None
        self.serial_reader: Optional[SerialReadThread] = None
        self.serial_lock = threading.Lock()
        self.line_queue: queue.Queue = queue.Queue()
        self.error_queue: queue.Queue = queue.Queue()
        self.connected = False
        self.firmware_ready = False
        self.active_port: Optional[str] = None
        self.settings_dirty = True
        self.config_acknowledged = False
        self.recording = False
        self.run_pending = False
        self.finishing = False
        self.result: Optional[HardwareResult] = None
        self.last_png_path: Optional[str] = None
        self.last_csv_path: Optional[str] = None
        self.last_ack_config: Optional[PanelConfig] = None
        self.current_run_config: Optional[PanelConfig] = None
        self.current_run_id = 0
        self.last_recorded_step = -1
        self.rows: List[List[float]] = []
        self.latest_revision = 0
        self.drawn_revision = -1
        self.pending_config: Optional[PanelConfig] = None
        self.pending_config_line = ""
        self.pending_config_token = 0
        self.pending_config_attempts = 0
        self.pending_config_auto = False
        self.go_attempts = 0
        self.restore_after_ready = False
        self.auto_reconnect_port: Optional[str] = None
        self.auto_reconnect_attempts = 0
        self.last_completion_summary = ""
        self.latest = {
            "time": 0.0,
            "theta": 0.0,
            "theta_dot": 0.0,
            "alpha": math.pi,
            "alpha_dot": 0.0,
            "pwm": 0.0,
            "mode": 0,
            "blend": 0.0,
            "pot": 0,
            "enc": 0,
        }
        self.settings_path = Path.home() / ".rip_dual_pid_hardware_student.json"
        self.build_ui()
        self.build_3d()
        self.load_settings()
        self.connect_dirty_signals()
        self.refresh_ports()
        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.timeout.connect(self.poll_queues)
        self.queue_timer.start(10)
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.timeout.connect(self.update_display)
        self.display_timer.start(40)
        self.handshake_timer = QtCore.QTimer(self)
        self.handshake_timer.timeout.connect(self.handshake_once)
        self.config_retry_timer = QtCore.QTimer(self)
        self.config_retry_timer.timeout.connect(self.retry_config)
        self.go_retry_timer = QtCore.QTimer(self)
        self.go_retry_timer.timeout.connect(self.retry_go)
        self.update_buttons()

    def dspin(
        self,
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
        step: float,
    ):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.setAlignment(QtCore.Qt.AlignRight)
        return widget

    def ispin(self, value: int, minimum: int, maximum: int, step: int = 1):
        widget = QtWidgets.QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.setAlignment(QtCore.Qt.AlignRight)
        return widget

    def build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(490)
        panel = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(panel)
        scroll.setWidget(panel)
        root.addWidget(scroll)

        connection = QtWidgets.QGroupBox("Serial Connection")
        connection_layout = QtWidgets.QGridLayout(connection)
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.toggle_connection)
        connection_layout.addWidget(self.port_combo, 0, 0, 1, 2)
        connection_layout.addWidget(self.refresh_button, 1, 0)
        connection_layout.addWidget(self.connect_button, 1, 1)
        self.connection_label = QtWidgets.QLabel("Disconnected")
        connection_layout.addWidget(self.connection_label, 2, 0, 1, 2)
        left.addWidget(connection)

        controller = QtWidgets.QGroupBox("Controller")
        controller_form = QtWidgets.QFormLayout(controller)
        self.duration = self.dspin(10.0, 0.1, 600.0, 3, 1.0)
        self.pwm_limit = self.dspin(150.0, 1.0, 255.0, 3, 1.0)
        self.swing_pwm = self.dspin(120.0, 0.0, 255.0, 3, 1.0)
        self.enter_deg = self.dspin(15.0, 0.1, 89.0, 3, 0.1)
        self.exit_deg = self.dspin(25.0, 0.2, 179.0, 3, 0.1)
        self.blend_alpha = self.dspin(0.18, 0.001, 1.0, 4, 0.01)
        controller_form.addRow("Duration / s:", self.duration)
        controller_form.addRow("PWM max:", self.pwm_limit)
        controller_form.addRow("Swing pump PWM:", self.swing_pwm)
        controller_form.addRow("PID enter / deg:", self.enter_deg)
        controller_form.addRow("PID exit / deg:", self.exit_deg)
        controller_form.addRow("Soft blend λ:", self.blend_alpha)
        left.addWidget(controller)

        gains = QtWidgets.QGroupBox("Dual PID Gains")
        gains_form = QtWidgets.QFormLayout(gains)
        self.kp_alpha = self.dspin(0.0, -5000.0, 5000.0, 4, 1.0)
        self.ki_alpha = self.dspin(0.0, -1000.0, 1000.0, 4, 1.0)
        self.kd_alpha = self.dspin(0.0, -2000.0, 2000.0, 4, 1.0)
        self.kp_theta = self.dspin(0.0, -2000.0, 2000.0, 4, 1.0)
        self.ki_theta = self.dspin(0.0, -1000.0, 1000.0, 4, 1.0)
        self.kd_theta = self.dspin(0.0, -2000.0, 2000.0, 4, 1.0)
        self.alpha_i_limit = self.dspin(0.50, 0.0, 100.0, 4, 0.1)
        self.theta_i_limit = self.dspin(1.00, 0.0, 100.0, 4, 0.1)
        gains_form.addRow("Pendulum Kp:", self.kp_alpha)
        gains_form.addRow("Pendulum Ki:", self.ki_alpha)
        gains_form.addRow("Pendulum Kd:", self.kd_alpha)
        gains_form.addRow("Arm Kp:", self.kp_theta)
        gains_form.addRow("Arm Ki:", self.ki_theta)
        gains_form.addRow("Arm Kd:", self.kd_theta)
        gains_form.addRow("Alpha integral limit:", self.alpha_i_limit)
        gains_form.addRow("Theta integral limit:", self.theta_i_limit)
        left.addWidget(gains)

        calibration = QtWidgets.QGroupBox("Sensor Calibration")
        calibration_form = QtWidgets.QFormLayout(calibration)
        self.raw_pot_label = QtWidgets.QLabel("--")
        self.raw_enc_label = QtWidgets.QLabel("--")
        self.pot_up = self.ispin(0, 0, 4095)
        self.pot_down = self.ispin(0, 0, 4095)
        self.cal_up_button = QtWidgets.QPushButton("Hold upright → Calibrate α=0")
        self.cal_down_button = QtWidgets.QPushButton("Hang down → Calibrate α=π")
        self.cal_up_button.clicked.connect(lambda: self.send_line("CALUP"))
        self.cal_down_button.clicked.connect(lambda: self.send_line("CALDOWN"))
        self.velocity_lpf = self.dspin(0.25, 0.001, 1.0, 4, 0.01)
        calibration_form.addRow("Current pot raw:", self.raw_pot_label)
        calibration_form.addRow("Current encoder:", self.raw_enc_label)
        calibration_form.addRow(self.cal_up_button)
        calibration_form.addRow(self.cal_down_button)
        calibration_form.addRow("PREF / upright raw:", self.pot_up)
        calibration_form.addRow("PDOWN / hanging raw:", self.pot_down)
        calibration_form.addRow("Velocity LPF:", self.velocity_lpf)
        left.addWidget(calibration)

        actions = QtWidgets.QGroupBox("Run")
        actions_layout = QtWidgets.QVBoxLayout(actions)
        self.save_button = QtWidgets.QPushButton("SAVE / Send Parameters to STM32")
        self.go_button = QtWidgets.QPushButton("GO")
        self.stop_button = QtWidgets.QPushButton("STOP")
        self.save_button.clicked.connect(self.save_and_send)
        self.go_button.clicked.connect(self.start_run)
        self.stop_button.clicked.connect(self.stop_run)
        actions_layout.addWidget(self.save_button)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.go_button)
        row.addWidget(self.stop_button)
        actions_layout.addLayout(row)
        self.status_label = QtWidgets.QLabel("Connect, calibrate, then SAVE.")
        self.status_label.setWordWrap(True)
        actions_layout.addWidget(self.status_label)
        self.state_label = QtWidgets.QLabel()
        actions_layout.addWidget(self.state_label)
        left.addWidget(actions)

        output = QtWidgets.QGroupBox("Result & Logging")
        output_layout = QtWidgets.QVBoxLayout(output)
        self.csv_checkbox = QtWidgets.QCheckBox("Generate CSV log")
        self.csv_checkbox.setChecked(True)
        self.output_dir = QtWidgets.QLineEdit(os.path.expanduser("~/rip_twin_logs"))
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self.choose_output_dir)
        show = QtWidgets.QPushButton("Show Last Result Curves")
        show.clicked.connect(self.show_result_dialog)
        output_layout.addWidget(self.csv_checkbox)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_dir, 1)
        output_row.addWidget(browse)
        output_layout.addLayout(output_row)
        output_layout.addWidget(show)
        left.addWidget(output)
        left.addStretch(1)
        self.right = QtWidgets.QWidget()
        root.addWidget(self.right, 1)

    def build_3d(self) -> None:
        layout = QtWidgets.QVBoxLayout(self.right)
        self.figure = Figure(figsize=(9, 7), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.axis.set_title("Real-time Hardware View", pad=2)
        self.axis.set_xlabel("X / m")
        self.axis.set_ylabel("Y / m")
        self.axis.set_zlabel("Z / m")
        limit = ARM_LENGTH + PEND_LENGTH + 0.05
        self.axis.set_xlim(-limit, limit)
        self.axis.set_ylim(-limit, limit)
        self.axis.set_zlim(-0.28, 0.38)
        try:
            self.axis.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        self.axis.view_init(elev=24, azim=-55)
        angle = np.linspace(0.0, 2.0 * math.pi, 80)
        self.axis.plot(
            MOTOR_RADIUS * np.cos(angle),
            MOTOR_RADIUS * np.sin(angle),
            MOTOR_HEIGHT * np.ones_like(angle),
            linewidth=2.0,
        )
        self.axis.plot(
            MOTOR_RADIUS * np.cos(angle),
            MOTOR_RADIUS * np.sin(angle),
            np.zeros_like(angle),
            linewidth=1.4,
        )
        for a in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
            x = MOTOR_RADIUS * math.cos(a)
            y = MOTOR_RADIUS * math.sin(a)
            self.axis.plot([x, x], [y, y], [0.0, MOTOR_HEIGHT], linewidth=1.0)
        self.axis.plot(
            ARM_LENGTH * np.cos(angle),
            ARM_LENGTH * np.sin(angle),
            ARM_Z * np.ones_like(angle),
            linestyle="--",
            linewidth=1.0,
        )
        self.arm_line, = self.axis.plot([], [], [], linewidth=6)
        self.pendulum_line, = self.axis.plot([], [], [], linewidth=5)
        self.joint_dot, = self.axis.plot([], [], [], marker="o", markersize=8)
        self.tip_dot, = self.axis.plot([], [], [], marker="o", markersize=10)
        self.tangent_line, = self.axis.plot([], [], [], linestyle=":", linewidth=2)
        self.reference_line, = self.axis.plot([], [], [], linestyle="--", linewidth=2.5)
        self.state_text = self.axis.text2D(
            0.03,
            0.89,
            "",
            transform=self.axis.transAxes,
            fontsize=11,
        )

    def numeric_widgets(self):
        return [
            self.duration,
            self.pwm_limit,
            self.swing_pwm,
            self.enter_deg,
            self.exit_deg,
            self.blend_alpha,
            self.kp_alpha,
            self.ki_alpha,
            self.kd_alpha,
            self.kp_theta,
            self.ki_theta,
            self.kd_theta,
            self.alpha_i_limit,
            self.theta_i_limit,
            self.pot_up,
            self.pot_down,
            self.velocity_lpf,
        ]

    def connect_dirty_signals(self) -> None:
        for widget in self.numeric_widgets():
            widget.valueChanged.connect(self.mark_dirty)

    def mark_dirty(self, *_args) -> None:
        self.settings_dirty = True
        self.config_acknowledged = False
        self.save_button.setText("SAVE / Send Parameters to STM32 *")
        self.update_buttons()

    def refresh_ports(self, preferred: Optional[str] = None) -> None:
        current = preferred or self.port_combo.currentData()
        self.port_combo.clear()
        for port in list(list_ports.comports()):
            label = f"{port.device} | {port.description}"
            self.port_combo.addItem(label, port.device)
        if current:
            index = self.port_combo.findData(current)
            if index >= 0:
                self.port_combo.setCurrentIndex(index)

    def toggle_connection(self) -> None:
        if self.connected:
            self.disconnect_serial(send_stop=True)
        else:
            port = self.port_combo.currentData()
            if not port:
                QtWidgets.QMessageBox.warning(self, "No port", "No serial port is selected.")
                return
            self.open_serial(str(port), automatic=False)

    def clear_queues(self) -> None:
        while True:
            try:
                self.line_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self.error_queue.get_nowait()
            except queue.Empty:
                break

    def open_serial(self, port: str, automatic: bool) -> bool:
        self.disconnect_serial(send_stop=False, update_status=False)
        self.clear_queues()
        try:
            serial_port = serial.Serial(
                port,
                BAUD,
                timeout=0.05,
                write_timeout=0.5,
            )
            serial_port.reset_input_buffer()
            serial_port.reset_output_buffer()
        except Exception as exc:
            if not automatic:
                QtWidgets.QMessageBox.critical(self, "Serial error", str(exc))
            return False
        self.serial = serial_port
        self.serial_reader = SerialReadThread(
            serial_port,
            self.line_queue,
            self.error_queue,
        )
        self.serial_reader.start()
        self.connected = True
        self.firmware_ready = False
        self.active_port = port
        self.config_acknowledged = False
        self.connection_label.setText(f"Opening: {port} @ {BAUD}")
        self.connect_button.setText("Disconnect")
        self.handshake_timer.start(350)
        QtCore.QTimer.singleShot(700, self.handshake_once)
        self.update_buttons()
        return True

    def disconnect_serial(
        self,
        send_stop: bool,
        update_status: bool = True,
    ) -> None:
        self.handshake_timer.stop()
        self.config_retry_timer.stop()
        self.go_retry_timer.stop()
        if self.serial is not None and send_stop:
            try:
                with self.serial_lock:
                    self.serial.write(b"STOP\n")
            except Exception:
                pass
        if self.serial_reader is not None:
            self.serial_reader.stop()
            self.serial_reader.join(timeout=0.3)
        if self.serial is not None:
            try:
                self.serial.close()
            except Exception:
                pass
        self.serial = None
        self.serial_reader = None
        self.connected = False
        self.firmware_ready = False
        self.pending_config = None
        self.pending_config_line = ""
        self.run_pending = False
        self.connection_label.setText("Disconnected")
        self.connect_button.setText("Connect")
        if update_status and not self.finishing:
            self.status_label.setText("Disconnected.")
        self.update_buttons()

    def send_line(self, line: str) -> bool:
        if not self.connected or self.serial is None:
            return False
        try:
            payload = (line.strip() + "\n").encode("ascii")
            with self.serial_lock:
                self.serial.write(payload)
            return True
        except Exception as exc:
            self.status_label.setText(f"Serial write error: {exc}")
            return False

    def handshake_once(self) -> None:
        if self.connected and not self.firmware_ready:
            self.send_line("HELLO")

    def poll_queues(self) -> None:
        try:
            error = self.error_queue.get_nowait()
        except queue.Empty:
            error = None
        if error and self.connected:
            port = self.active_port
            self.disconnect_serial(send_stop=False)
            self.status_label.setText(f"Serial disconnected: {error}")
            if self.recording or self.run_pending:
                self.recording = False
                self.run_pending = False
                self.finalize_result()
            if port:
                self.schedule_auto_reconnect(port)
            return
        deadline = time.perf_counter() + 0.006
        count = 0
        while count < 2000 and time.perf_counter() < deadline:
            try:
                line = self.line_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_line(line)
            count += 1

    def handle_line(self, line: str) -> None:
        if line.startswith("READY,"):
            parts = line.split(",")
            if len(parts) < 3 or parts[2] != "2":
                self.status_label.setText("Firmware version mismatch. Flash v2 firmware.")
                return
            self.firmware_ready = True
            self.handshake_timer.stop()
            self.connection_label.setText(f"Connected: {self.active_port} @ {BAUD}")
            self.send_line("STATUS")
            if self.restore_after_ready and self.last_ack_config is not None:
                self.begin_config_send(self.last_ack_config, automatic=True)
            else:
                self.status_label.setText("STM32 firmware ready. Calibrate and SAVE.")
            self.update_buttons()
            return
        if line.startswith("ACK,CONFIG,"):
            parts = line.split(",")
            if len(parts) != 3:
                return
            try:
                token = int(parts[2])
            except ValueError:
                return
            if self.pending_config is None or token != self.pending_config_token:
                return
            config = self.pending_config
            automatic = self.pending_config_auto
            self.config_retry_timer.stop()
            self.pending_config = None
            self.pending_config_line = ""
            self.pending_config_attempts = 0
            self.pending_config_auto = False
            self.settings_dirty = False
            self.config_acknowledged = True
            self.last_ack_config = config
            self.save_button.setText("SAVE / Send Parameters to STM32")
            self.save_local_settings()
            if automatic:
                self.restore_after_ready = False
                text = "Serial reconnected and parameters restored automatically."
                if self.last_completion_summary:
                    text += "\n" + self.last_completion_summary
                self.status_label.setText(text)
            else:
                self.status_label.setText("Parameters saved locally and active on STM32.")
            self.update_buttons()
            return
        if line.startswith("CAL,"):
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    value = int(parts[2])
                except ValueError:
                    return
                if parts[1] == "UP":
                    self.pot_up.setValue(value)
                    self.status_label.setText(f"Upright potentiometer zero captured: {value}")
                elif parts[1] == "DOWN":
                    self.pot_down.setValue(value)
                    self.status_label.setText(f"Hanging-down reference captured: {value}")
            return
        if line.startswith("ARMED,"):
            parts = line.split(",")
            if len(parts) != 2:
                return
            try:
                run_id = int(parts[1])
            except ValueError:
                return
            if run_id != self.current_run_id:
                return
            self.go_retry_timer.stop()
            self.rows.clear()
            self.last_recorded_step = -1
            self.recording = True
            self.run_pending = False
            self.latest["time"] = 0.0
            self.status_label.setText("Firmware control loop running at 200 Hz.")
            self.update_buttons()
            return
        if line.startswith("DONE,") or line.startswith("STOPPED,"):
            parts = line.split(",")
            if len(parts) != 2:
                return
            try:
                run_id = int(parts[1])
            except ValueError:
                return
            if run_id != self.current_run_id:
                return
            message = (
                "Configured duration completed."
                if line.startswith("DONE,")
                else "Run stopped."
            )
            self.complete_run(message)
            return
        if line.startswith("ERR,"):
            if line == "ERR,CONFIG_VALUE":
                self.config_retry_timer.stop()
                self.pending_config = None
                self.pending_config_line = ""
            self.status_label.setText(line)
            self.update_buttons()
            return
        if line.startswith("STATUS,"):
            return
        if line.startswith("MON,"):
            self.handle_monitor(line)
            return
        if line.startswith("TEL,"):
            self.handle_telemetry(line)

    def handle_monitor(self, line: str) -> None:
        parts = line.split(",")
        if len(parts) != 7:
            return
        try:
            self.latest.update(
                {
                    "theta": float(parts[1]),
                    "theta_dot": float(parts[2]),
                    "alpha": float(parts[3]),
                    "alpha_dot": float(parts[4]),
                    "pwm": 0.0,
                    "mode": 0,
                    "blend": 0.0,
                    "pot": int(parts[5]),
                    "enc": int(parts[6]),
                }
            )
        except ValueError:
            return
        self.latest_revision += 1

    def handle_telemetry(self, line: str) -> None:
        parts = line.split(",")
        if len(parts) != 12:
            return
        try:
            run_id = int(parts[1])
            step = int(parts[2])
            values = {
                "time": float(step) * CONTROL_DT,
                "theta": float(parts[3]),
                "theta_dot": float(parts[4]),
                "alpha": float(parts[5]),
                "alpha_dot": float(parts[6]),
                "pwm": float(parts[7]),
                "mode": int(parts[8]),
                "blend": float(parts[9]),
                "pot": int(parts[10]),
                "enc": int(parts[11]),
            }
        except ValueError:
            return
        self.latest.update(values)
        self.latest_revision += 1
        if (
            self.recording
            and run_id == self.current_run_id
            and step > self.last_recorded_step
        ):
            self.rows.append(
                [
                    values["time"],
                    values["theta"],
                    values["theta_dot"],
                    values["alpha"],
                    values["alpha_dot"],
                    values["pwm"],
                    values["mode"],
                    values["blend"],
                ]
            )
            self.last_recorded_step = step

    def read_config(self) -> PanelConfig:
        config = PanelConfig(
            duration=float(self.duration.value()),
            kp_alpha=float(self.kp_alpha.value()),
            ki_alpha=float(self.ki_alpha.value()),
            kd_alpha=float(self.kd_alpha.value()),
            kp_theta=float(self.kp_theta.value()),
            ki_theta=float(self.ki_theta.value()),
            kd_theta=float(self.kd_theta.value()),
            pwm_limit=float(self.pwm_limit.value()),
            swing_pwm=float(self.swing_pwm.value()),
            enter_deg=float(self.enter_deg.value()),
            exit_deg=float(self.exit_deg.value()),
            blend_alpha=float(self.blend_alpha.value()),
            alpha_i_limit=float(self.alpha_i_limit.value()),
            theta_i_limit=float(self.theta_i_limit.value()),
            velocity_lpf=float(self.velocity_lpf.value()),
            pot_up=int(self.pot_up.value()),
            pot_down=int(self.pot_down.value()),
        )
        if config.exit_deg <= config.enter_deg:
            raise ValueError("PID exit angle must be greater than PID enter angle.")
        if config.swing_pwm > config.pwm_limit:
            raise ValueError("Swing pump PWM cannot exceed PWM max.")
        if config.pot_up == config.pot_down:
            raise ValueError("Upright and hanging-down potentiometer values must differ.")
        return config

    def config_line(self, config: PanelConfig, token: int) -> str:
        values = [
            "CONFIG",
            "2",
            str(token),
            f"{config.duration:.9g}",
            f"{config.kp_alpha:.9g}",
            f"{config.ki_alpha:.9g}",
            f"{config.kd_alpha:.9g}",
            f"{config.kp_theta:.9g}",
            f"{config.ki_theta:.9g}",
            f"{config.kd_theta:.9g}",
            f"{config.pwm_limit:.9g}",
            f"{config.swing_pwm:.9g}",
            f"{config.kick_time:.9g}",
            f"{config.enter_deg:.9g}",
            f"{config.exit_deg:.9g}",
            f"{config.blend_alpha:.9g}",
            f"{config.alpha_i_limit:.9g}",
            f"{config.theta_i_limit:.9g}",
            f"{config.velocity_lpf:.9g}",
            str(config.pot_up),
            str(config.pot_down),
            f"{config.theta_rad_per_count:.12g}",
            str(config.motor_sign),
        ]
        return ",".join(values)

    def save_and_send(self) -> None:
        if not self.connected or not self.firmware_ready:
            QtWidgets.QMessageBox.warning(self, "Not ready", "Connect the STM32 and wait for READY.")
            return
        try:
            config = self.read_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid setting", str(exc))
            return
        self.begin_config_send(config, automatic=False)

    def begin_config_send(self, config: PanelConfig, automatic: bool) -> None:
        self.pending_config_token = (self.pending_config_token + 1) % 2000000000
        if self.pending_config_token <= 0:
            self.pending_config_token = 1
        self.pending_config = config
        self.pending_config_line = self.config_line(config, self.pending_config_token)
        self.pending_config_attempts = 0
        self.pending_config_auto = automatic
        self.config_acknowledged = False
        self.retry_config()
        self.config_retry_timer.start(300)
        self.status_label.setText(
            "Restoring parameters after reconnect..."
            if automatic
            else "Sending parameters to STM32..."
        )
        self.update_buttons()

    def retry_config(self) -> None:
        if self.pending_config is None:
            self.config_retry_timer.stop()
            return
        if not self.connected or not self.firmware_ready:
            return
        if self.pending_config_attempts >= 12:
            self.config_retry_timer.stop()
            self.status_label.setText("SAVE failed after automatic retries. Check the serial connection.")
            self.pending_config = None
            self.pending_config_line = ""
            self.update_buttons()
            return
        self.pending_config_attempts += 1
        self.send_line(self.pending_config_line)

    def start_run(self) -> None:
        if not self.connected or not self.firmware_ready:
            return
        if self.settings_dirty or not self.config_acknowledged:
            QtWidgets.QMessageBox.warning(
                self,
                "Save required",
                "Click SAVE and wait for the STM32 acknowledgement before GO.",
            )
            return
        self.current_run_id = int(time.monotonic_ns() // 1000000) % 2000000000
        if self.current_run_id <= 0:
            self.current_run_id = 1
        self.current_run_config = self.read_config()
        self.rows.clear()
        self.result = None
        self.recording = False
        self.run_pending = True
        self.last_recorded_step = -1
        self.go_attempts = 0
        self.retry_go()
        self.go_retry_timer.start(300)
        self.status_label.setText("Starting firmware control loop...")
        self.update_buttons()

    def retry_go(self) -> None:
        if not self.run_pending:
            self.go_retry_timer.stop()
            return
        if not self.connected or not self.firmware_ready:
            return
        if self.go_attempts >= 8:
            self.go_retry_timer.stop()
            self.run_pending = False
            self.status_label.setText("GO was not acknowledged. Check the serial connection.")
            self.update_buttons()
            return
        self.go_attempts += 1
        self.send_line(f"GO,{self.current_run_id}")

    def stop_run(self) -> None:
        if not self.connected:
            return
        self.send_line(f"STOP,{self.current_run_id}")
        QtCore.QTimer.singleShot(80, lambda: self.send_line(f"STOP,{self.current_run_id}"))
        QtCore.QTimer.singleShot(180, lambda: self.send_line(f"STOP,{self.current_run_id}"))

    def complete_run(self, message: str) -> None:
        if self.finishing:
            return
        self.finishing = True
        self.recording = False
        self.run_pending = False
        self.go_retry_timer.stop()
        port = self.active_port
        self.disconnect_serial(send_stop=False, update_status=False)
        summary = self.finalize_result()
        self.last_completion_summary = message
        if summary:
            self.last_completion_summary += "\n" + summary
        if port:
            self.restore_after_ready = True
            self.status_label.setText(self.last_completion_summary + "\nReconnecting serial...")
            self.schedule_auto_reconnect(port)
        else:
            self.status_label.setText(self.last_completion_summary)
        self.finishing = False
        self.update_buttons()

    def schedule_auto_reconnect(self, port: str) -> None:
        self.auto_reconnect_port = port
        self.auto_reconnect_attempts = 0
        QtCore.QTimer.singleShot(700, self.attempt_auto_reconnect)

    def attempt_auto_reconnect(self) -> None:
        if self.connected or not self.auto_reconnect_port:
            return
        port = self.auto_reconnect_port
        self.auto_reconnect_attempts += 1
        self.refresh_ports(port)
        if self.open_serial(port, automatic=True):
            self.auto_reconnect_port = None
            return
        if self.auto_reconnect_attempts < 20:
            self.status_label.setText(
                self.last_completion_summary
                + f"\nWaiting for serial reconnect ({self.auto_reconnect_attempts}/20)..."
            )
            QtCore.QTimer.singleShot(1000, self.attempt_auto_reconnect)
        else:
            self.status_label.setText(
                self.last_completion_summary
                + "\nAutomatic reconnect failed. Select the port and click Connect."
            )
            self.auto_reconnect_port = None

    def update_buttons(self) -> None:
        busy = self.recording or self.run_pending or self.pending_config is not None
        self.go_button.setEnabled(
            self.connected
            and self.firmware_ready
            and self.config_acknowledged
            and not self.settings_dirty
            and not busy
        )
        self.stop_button.setEnabled(self.connected and (self.recording or self.run_pending))
        self.save_button.setEnabled(
            self.connected
            and self.firmware_ready
            and not self.recording
            and not self.run_pending
            and self.pending_config is None
        )
        self.cal_up_button.setEnabled(
            self.connected and self.firmware_ready and not busy
        )
        self.cal_down_button.setEnabled(
            self.connected and self.firmware_ready and not busy
        )

    def update_display(self) -> None:
        self.raw_pot_label.setText(str(self.latest["pot"]))
        self.raw_enc_label.setText(str(self.latest["enc"]))
        self.state_label.setText(
            f"theta: {float(self.latest['theta']):+.5f} rad\n"
            f"alpha: {float(self.latest['alpha']):+.5f} rad\n"
            f"theta_dot: {float(self.latest['theta_dot']):+.5f} rad/s\n"
            f"alpha_dot: {float(self.latest['alpha_dot']):+.5f} rad/s\n"
            f"PWM: {float(self.latest['pwm']):+.0f}\n"
            f"mode: {MODE_NAMES.get(int(self.latest['mode']), 'UNKNOWN')}"
        )
        if self.latest_revision == self.drawn_revision:
            return
        self.drawn_revision = self.latest_revision
        theta = float(self.latest["theta"])
        alpha = float(self.latest["alpha"])
        center, joint, tip, reference, tangent = rip_points(theta, alpha)
        set_line3d(self.arm_line, center, joint)
        set_line3d(self.pendulum_line, joint, tip)
        set_point3d(self.joint_dot, joint)
        set_point3d(self.tip_dot, tip)
        set_line3d(self.tangent_line, joint - 0.12 * tangent, joint + 0.12 * tangent)
        set_line3d(self.reference_line, joint, reference)
        self.state_text.set_text(
            f"t = {float(self.latest['time']):6.3f} s\n"
            f"theta = {theta:+.4f} rad\n"
            f"alpha = {alpha:+.4f} rad\n"
            f"PWM = {float(self.latest['pwm']):+.0f}"
        )
        self.canvas.draw_idle()

    def rows_to_result(self) -> Optional[HardwareResult]:
        if len(self.rows) < 2:
            return None
        array = np.asarray(self.rows, dtype=np.float64)
        finite = np.all(np.isfinite(array), axis=1)
        array = array[finite]
        if array.shape[0] < 2:
            return None
        keep = np.ones(array.shape[0], dtype=bool)
        keep[1:] = np.diff(array[:, 0]) > 0.0
        array = array[keep]
        if array.shape[0] < 2:
            return None
        time_values = array[:, 0] - array[0, 0]
        time_values[0] = 0.0
        return HardwareResult(
            time=time_values,
            theta=array[:, 1],
            theta_dot=array[:, 2],
            alpha=array[:, 3],
            alpha_dot=array[:, 4],
            pwm=array[:, 5],
            mode=array[:, 6].astype(np.int32),
            blend=array[:, 7],
        )

    def finalize_result(self) -> str:
        result = self.rows_to_result()
        if result is None:
            return "No valid run samples were received."
        self.result = result
        duration = (
            self.current_run_config.duration
            if self.current_run_config is not None
            else result.time[-1]
        )
        output_dir = self.output_dir.text().strip() or "~/rip_twin_logs"
        png_path, csv_path = default_output_paths(output_dir, duration)
        figure = build_result_figure(result)
        figure.savefig(png_path, dpi=200, bbox_inches="tight")
        figure.clear()
        self.last_png_path = png_path
        self.last_csv_path = None
        if self.csv_checkbox.isChecked():
            save_result_csv(result, csv_path)
            self.last_csv_path = csv_path
        metrics = result_metrics(result)
        if math.isnan(metrics["stable_start_time"]):
            stable_text = "Final stable stage was not reached."
        else:
            stable_text = (
                f"Stable from {metrics['stable_start_time']:.3f} s; "
                f"mean|alpha|={metrics['alpha_abs_mean']:.6f} rad; "
                f"std|alpha|={metrics['alpha_abs_std']:.6f} rad; "
                f"mean|PWM|={metrics['pwm_abs_mean']:.3f}; "
                f"std|PWM|={metrics['pwm_abs_std']:.3f}."
            )
        saved = [png_path]
        if self.last_csv_path:
            saved.append(self.last_csv_path)
        return stable_text + "\nSaved:\n" + "\n".join(saved)

    def show_result_dialog(self) -> None:
        if self.result is None:
            self.status_label.setText("No completed result yet.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"RIP Hardware Result | {self.result.time[-1]:.3f} s")
        dialog.resize(1120, 860)
        layout = QtWidgets.QVBoxLayout(dialog)
        figure = build_result_figure(self.result)
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, alignment=QtCore.Qt.AlignRight)
        dialog.exec_()

    def choose_output_dir(self) -> None:
        start = self.output_dir.text().strip() or os.path.expanduser("~/rip_twin_logs")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose result directory",
            os.path.expanduser(start),
        )
        if path:
            self.output_dir.setText(path)

    def save_local_settings(self) -> None:
        try:
            config = self.read_config()
            payload = asdict(config)
            payload["output_dir"] = self.output_dir.text()
            payload["save_csv"] = self.csv_checkbox.isChecked()
            self.settings_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def load_settings(self) -> None:
        if not self.settings_path.is_file():
            return
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:
            return
        mapping = {
            "duration": self.duration,
            "kp_alpha": self.kp_alpha,
            "ki_alpha": self.ki_alpha,
            "kd_alpha": self.kd_alpha,
            "kp_theta": self.kp_theta,
            "ki_theta": self.ki_theta,
            "kd_theta": self.kd_theta,
            "pwm_limit": self.pwm_limit,
            "swing_pwm": self.swing_pwm,
            "enter_deg": self.enter_deg,
            "exit_deg": self.exit_deg,
            "blend_alpha": self.blend_alpha,
            "alpha_i_limit": self.alpha_i_limit,
            "theta_i_limit": self.theta_i_limit,
            "velocity_lpf": self.velocity_lpf,
            "pot_up": self.pot_up,
            "pot_down": self.pot_down,
        }
        for key, widget in mapping.items():
            if key in data:
                widget.setValue(data[key])
        if "output_dir" in data:
            self.output_dir.setText(data["output_dir"])
        self.csv_checkbox.setChecked(bool(data.get("save_csv", True)))

    def closeEvent(self, event) -> None:
        self.disconnect_serial(send_stop=True, update_status=False)
        event.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
