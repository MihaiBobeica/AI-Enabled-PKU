#!/usr/bin/env python3
"""Hardware panel for energy-swing-up plus distilled PPO upright control.

The GUI mirrors the DQN deployment panel: serial connection, model/header or
run-folder selection, CRC-checked SAVE upload, calibration, GO/STOP, 200 Hz
telemetry, live 3-D view, CSV logging and result figures.

The accepted model is the generated compact7 PPO header with actor
7->64->64->1 and internal observation normalization.  The panel parses the
header and uploads the original float32 student network to STM32 RAM.  Far from
upright the firmware uses MPC-style phase/energy pumping; near upright it
blends to PPO using the fixed 5 ms Luenberger observer.
"""
from __future__ import annotations

import csv
import json
import math
import os
import queue
import sys
import threading
import time
import zlib
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

from rip_ppo_sim_test import PPOModel, load_model, rip_points, set_line3d, set_point3d

BAUD = 921600
CONTROL_DT = 0.005
ARM_LENGTH = 0.18
PEND_LENGTH = 0.24
MOTOR_RADIUS = 0.035
MOTOR_HEIGHT = 0.08
ARM_Z = MOTOR_HEIGHT
PROTOCOL_VERSION = 11
MODE_NAMES = {0: "DISABLED", 1: "ENERGY_SWING", 2: "BLEND", 3: "PPO_BALANCE"}


@dataclass
class PanelConfig:
    duration: float = 30.0
    safety_pwm_limit: float = 150.0
    swing_pwm: float = 120.0
    kick_time: float = 0.10
    ppo_enter_deg: float = 15.0
    ppo_exit_deg: float = 25.0
    blend_alpha: float = 0.18
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
    action_norm: np.ndarray
    policy_raw: np.ndarray


def stable_phase_start_index(result: HardwareResult, threshold_deg: float = 15.0) -> Optional[int]:
    if result.alpha.size == 0:
        return None
    inside = np.abs(result.alpha) <= math.radians(threshold_deg)
    outside = np.flatnonzero(~inside)
    start = int(outside[-1] + 1) if outside.size else 0
    return start if start < len(inside) else None


def result_metrics(result: HardwareResult) -> Dict[str, float]:
    index = stable_phase_start_index(result)
    output = {
        "stable_start_time": math.nan, "stable_duration": 0.0,
        "alpha_abs_mean": math.nan, "alpha_abs_std": math.nan,
        "pwm_abs_mean": math.nan, "pwm_abs_std": math.nan,
        "max_abs_theta": float(np.max(np.abs(result.theta))) if result.theta.size else math.nan,
    }
    if index is not None:
        output.update(
            stable_start_time=float(result.time[index]),
            stable_duration=float(result.time[-1] - result.time[index]),
            alpha_abs_mean=float(np.mean(np.abs(result.alpha[index:]))),
            alpha_abs_std=float(np.std(np.abs(result.alpha[index:]))),
            pwm_abs_mean=float(np.mean(np.abs(result.pwm[index:]))),
            pwm_abs_std=float(np.std(np.abs(result.pwm[index:]))),
        )
    return output


def build_result_figure(result: HardwareResult, title_suffix: str = "Hardware Hybrid PPO") -> Figure:
    metrics = result_metrics(result)
    fig = Figure(figsize=(10.5, 7.6), tight_layout=True)
    axs = fig.subplots(2, 2)
    ax_alpha, ax_theta, ax_pwm, ax_hist = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    t, alpha, theta, pwm = result.time, result.alpha, result.theta, result.pwm
    fig.suptitle(f"Rotary Inverted Pendulum {title_suffix} Response", fontsize=15, fontweight="bold")
    ax_alpha.plot(t, alpha, linewidth=1.8, label=r"$\alpha$")
    ax_alpha.axhline(0, linewidth=1, linestyle="--")
    for sign in (-1, 1): ax_alpha.axhline(sign * math.radians(15), linewidth=0.8, linestyle=":")
    ax_alpha.set_title(r"Pendulum Angle $\alpha(t)$", fontsize=12, fontweight="bold")
    ax_alpha.set_xlabel("Time / s"); ax_alpha.set_ylabel(r"$\alpha$ / rad"); ax_alpha.grid(True, linestyle="--", linewidth=0.6, alpha=0.55); ax_alpha.legend(loc="lower right")
    index = stable_phase_start_index(result)
    if index is None:
        alpha_text = "Stable phase: not reached\ncriterion: |alpha| <= 15 deg until the end"
    else:
        alpha_text = (rf"$\mathrm{{mean}}(|\alpha|)$ = {metrics['alpha_abs_mean']:.6f} rad" "\n"
                      rf"$\mathrm{{std}}(|\alpha|)$ = {metrics['alpha_abs_std']:.6f} rad" "\n"
                      f"stable from t = {metrics['stable_start_time']:.3f} s")
        ax_alpha.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_alpha.text(0.98, 0.96, alpha_text, transform=ax_alpha.transAxes, ha="right", va="top", fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"))
    ax_theta.plot(t, theta, linewidth=1.8, label=r"$\theta$"); ax_theta.axhline(0, linewidth=1, linestyle="--")
    ax_theta.set_title(r"Rotary Arm Angle $\theta(t)$", fontsize=12, fontweight="bold"); ax_theta.set_xlabel("Time / s"); ax_theta.set_ylabel(r"$\theta$ / rad")
    ax_theta.grid(True, linestyle="--", linewidth=0.6, alpha=0.55); ax_theta.legend(loc="upper right")
    ax_pwm.plot(t, pwm, linewidth=1.5, label="PWM"); ax_pwm.axhline(0, linewidth=1, linestyle="--")
    ax_pwm.set_title("Control Input PWM(t)", fontsize=12, fontweight="bold"); ax_pwm.set_xlabel("Time / s"); ax_pwm.set_ylabel("PWM")
    lim = max(160.0, float(np.max(np.abs(pwm))) * 1.1); ax_pwm.set_ylim(-lim, lim); ax_pwm.grid(True, linestyle="--", linewidth=0.6, alpha=0.55); ax_pwm.legend(loc="lower right")
    if index is None: pwm_text = "Stable phase: not reached"
    else:
        pwm_text = rf"$\mathrm{{mean}}(|PWM|)$ = {metrics['pwm_abs_mean']:.3f}" "\n" rf"$\mathrm{{std}}(|PWM|)$ = {metrics['pwm_abs_std']:.3f}"
        ax_pwm.axvspan(metrics["stable_start_time"], t[-1], alpha=0.10)
    ax_pwm.text(0.98, 0.96, pwm_text, transform=ax_pwm.transAxes, ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.35"))
    if index is None: ax_hist.text(0.5, 0.5, "No final stable phase", transform=ax_hist.transAxes, ha="center", va="center")
    else: ax_hist.hist(pwm[index:], bins=np.arange(-255, 271, 15), edgecolor="black", linewidth=0.45)
    ax_hist.axvline(0, linewidth=1, linestyle="--"); ax_hist.set_xlim(-255,255); ax_hist.set_title("Stable-stage PWM Distribution", fontsize=12, fontweight="bold")
    ax_hist.set_xlabel("PWM"); ax_hist.set_ylabel("Count"); ax_hist.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    xmax=max(float(t[-1]),0.1)
    for ax in (ax_alpha,ax_theta,ax_pwm): ax.set_xlim(0,xmax)
    for ax in (ax_alpha,ax_theta,ax_pwm,ax_hist): ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.tick_params(direction="in")
    return fig


def save_result_csv(result: HardwareResult, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["time_s","theta_rad","theta_dot_rad_s","alpha_rad","alpha_dot_rad_s","pwm","mode","blend","action_norm","policy_raw"])
        writer.writerows(zip(result.time,result.theta,result.theta_dot,result.alpha,result.alpha_dot,result.pwm,result.mode,result.blend,result.action_norm,result.policy_raw))


def default_output_paths(output_dir: str, duration: float) -> Tuple[str,str]:
    directory=Path(output_dir).expanduser().resolve(); directory.mkdir(parents=True,exist_ok=True); stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    tag=f"{duration:.3f}".rstrip("0").rstrip(".").replace(".","p")
    return str(directory/f"rip_ppo_hybrid_hardware_{tag}s_{stamp}.png"), str(directory/f"rip_ppo_hybrid_hardware_{tag}s_{stamp}.csv")


class SerialReadThread(threading.Thread):
    """Read complete ASCII lines and route model-protocol replies directly.

    Model upload runs in a worker thread. Routing its ACK/ERR lines here avoids
    depending on the GUI timer to relay replies while an upload is active.
    """
    def __init__(
        self,
        port: serial.Serial,
        line_queue: queue.Queue,
        error_queue: queue.Queue,
        model_reply_queue: queue.Queue,
        generation: int,
    ):
        super().__init__(daemon=True)
        self.port = port
        self.line_queue = line_queue
        self.error_queue = error_queue
        self.model_reply_queue = model_reply_queue
        self.generation = int(generation)
        self.stop_event = threading.Event()

    def run(self):
        buffer = bytearray()
        while not self.stop_event.is_set():
            try:
                count = self.port.in_waiting
                data = self.port.read(count if count > 0 else 1)
            except Exception as exc:
                if not self.stop_event.is_set():
                    self.error_queue.put((self.generation, str(exc)))
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
                if not line:
                    continue
                if line.startswith("ACK,MODEL") or line.startswith("ERR,MODEL"):
                    self.model_reply_queue.put((self.generation, line))
                else:
                    self.line_queue.put((self.generation, line))

    def stop(self):
        self.stop_event.set()


class ModelUploadThread(QtCore.QThread):
    """Reliable ASCII-hex model uploader.

    Protocol v11 sends each compact7 float32 model block in one self-contained ASCII line.  This
    avoids switching the STM32 parser from line mode to raw-binary mode, which
    proved unreliable on some macOS/ST-Link virtual-serial combinations.  Every
    line carries its offset, length and CRC32; the STM32 acknowledges a block
    only after decoding, checking and committing it.  Re-sending an already
    committed block is idempotent.
    """

    progress = QtCore.pyqtSignal(int, str)
    completed = QtCore.pyqtSignal(bool, str, object)
    CHUNK_BYTES = 96
    CHUNK_RETRIES = 10
    READY_SETTLE_SECONDS = 0.050
    INTER_CHUNK_SECONDS = 0.002

    def __init__(self, panel: "MainWindow", model: PPOModel, token: int, config: PanelConfig):
        super().__init__(panel)
        self.panel = panel
        self.model = model
        self.token = token
        self.config = config
        self.generation = int(panel.connection_generation)
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def ensure_active(self) -> None:
        if self.cancel_event.is_set():
            raise RuntimeError("Upload cancelled")
        if not self.panel.connected or self.panel.connection_generation != self.generation:
            raise ConnectionError("Serial connection changed during upload")

    @staticmethod
    def build_blob(model: PPOModel) -> bytes:
        blob = model.float_blob()
        expected = (64 * 7 + 64 + 64 * 64 + 64 + 1 * 64 + 1 + 7 + 7 + 1 + 1) * 4
        if len(blob) != expected:
            raise AssertionError(f"Unexpected float32 model blob size {len(blob)} != {expected}")
        return blob

    def wait_model_line(self, prefixes: tuple[str, ...], timeout: float) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.ensure_active()
            try:
                generation, line = self.panel.model_reply_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if generation != self.generation:
                continue
            if line.startswith(prefixes) or line.startswith("ERR,MODEL"):
                return line
        raise TimeoutError("No STM32 reply for " + " or ".join(prefixes))

    def command(
        self,
        line: str,
        expected_prefix: str,
        *,
        retries: int = 5,
        timeout: float = 5.0,
    ) -> str:
        last: Optional[Exception] = None
        for _ in range(retries):
            self.ensure_active()
            if not self.panel.send_line(line, generation=self.generation):
                raise ConnectionError("Serial command write failed")
            try:
                reply = self.wait_model_line((expected_prefix,), timeout)
            except TimeoutError as exc:
                last = exc
                continue
            if reply.startswith(expected_prefix):
                return reply
            last = RuntimeError(reply)
        raise last or TimeoutError(expected_prefix)

    def upload_chunk(self, start: int, block: bytes) -> None:
        token = self.token
        end = start + len(block)
        chunk_crc = zlib.crc32(block) & 0xFFFFFFFF
        expected_lead = f"ACK,MODEL_HEX_CHUNK,{token},"
        line = (
            f"MODEL_HEX_CHUNK,{PROTOCOL_VERSION},{token},{start},"
            f"{len(block)},{chunk_crc:08x},{block.hex()}"
        )
        last_error: Optional[Exception] = None

        for _ in range(self.CHUNK_RETRIES):
            self.ensure_active()
            if not self.panel.send_line(line, generation=self.generation):
                raise ConnectionError(f"Chunk line write failed at byte {start}")
            try:
                reply = self.wait_model_line((expected_lead,), timeout=5.0)
            except TimeoutError as exc:
                last_error = exc
                continue
            if reply.startswith("ERR,MODEL"):
                last_error = RuntimeError(reply)
                continue
            parts = reply.split(",")
            try:
                ok = (
                    len(parts) == 7
                    and parts[0] == "ACK"
                    and parts[1] == "MODEL_HEX_CHUNK"
                    and int(parts[2]) == token
                    and int(parts[3]) == start
                    and int(parts[4]) == end
                    and int(parts[5]) == len(block)
                    and int(parts[6], 16) == chunk_crc
                )
            except ValueError:
                ok = False
            if ok:
                return
            last_error = RuntimeError(f"Unexpected chunk reply: {reply}")

        raise last_error or TimeoutError(f"Chunk {start}:{end} was not acknowledged")

    def run(self) -> None:
        try:
            while True:
                try:
                    self.panel.model_reply_queue.get_nowait()
                except queue.Empty:
                    break

            blob = self.build_blob(self.model)
            crc = zlib.crc32(blob) & 0xFFFFFFFF
            token = self.token
            begin = (
                f"MODEL_HEX_BEGIN,{PROTOCOL_VERSION},{token},{len(blob)},"
                f"{self.model.digest},{crc:08x}"
            )
            ready_prefix = f"ACK,MODEL_HEX_READY,{token},{len(blob)}"
            chunk_count = (len(blob) + self.CHUNK_BYTES - 1) // self.CHUNK_BYTES
            self.progress.emit(
                0,
                f"Preparing reliable ASCII float32 upload: {len(blob)} bytes in {chunk_count} chunks",
            )
            self.command(begin, ready_prefix, retries=8, timeout=6.0)
            time.sleep(self.READY_SETTLE_SECONDS)

            sent = 0
            started = time.monotonic()
            for chunk_index, start in enumerate(
                range(0, len(blob), self.CHUNK_BYTES), start=1
            ):
                block = blob[start : start + self.CHUNK_BYTES]
                self.upload_chunk(start, block)
                time.sleep(self.INTER_CHUNK_SECONDS)
                sent += len(block)
                value = int(round(100.0 * sent / len(blob)))
                elapsed = max(time.monotonic() - started, 1.0e-6)
                rate = sent / elapsed / 1024.0
                self.progress.emit(
                    value,
                    f"STM32 confirmed chunk {chunk_index}/{chunk_count}: "
                    f"{sent}/{len(blob)} bytes ({value}%, {rate:.1f} KiB/s)",
                )

            end_line = f"MODEL_HEX_END,{PROTOCOL_VERSION},{token},{crc:08x}"
            done_prefix = f"ACK,MODEL_HEX_DONE,{token},"
            reply = self.command(end_line, done_prefix, retries=8, timeout=10.0)
            parts = reply.split(",")
            if len(parts) != 5 or parts[3] != self.model.digest:
                raise RuntimeError(f"Unexpected model completion reply: {reply}")
            received_crc = parts[4].lower()
            if received_crc != f"{crc:08x}":
                raise RuntimeError(
                    f"STM32 CRC reply mismatch: host={crc:08x}, device={received_crc}"
                )
            self.completed.emit(
                True,
                f"ASCII model upload verified (CRC32 {crc:08x}, ID {self.model.digest})",
                self.model,
            )
        except Exception as exc:
            try:
                self.panel.send_line(
                    f"MODEL_HEX_ABORT,{PROTOCOL_VERSION},{self.token}", generation=self.generation
                )
            except Exception:
                pass
            self.completed.emit(False, str(exc), self.model)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("RIP Hardware Hybrid PPO Control Panel - Stable Serial v11"); self.resize(1240,820)
        self.serial: Optional[serial.Serial]=None; self.serial_reader: Optional[SerialReadThread]=None; self.serial_lock=threading.RLock(); self.connection_generation=0
        self.line_queue=queue.Queue(); self.error_queue=queue.Queue(); self.model_reply_queue=queue.Queue()
        self.connected=False; self.firmware_ready=False; self.active_port=None; self.settings_dirty=True; self.config_acknowledged=False; self.model_acknowledged=False
        self.upload_in_progress=False; self.upload_thread: Optional[ModelUploadThread]=None; self.loaded_model: Optional[PPOModel]=None; self.last_model_source=""
        self.recording=False; self.run_pending=False; self.finishing=False; self.result=None; self.last_png_path=None; self.last_csv_path=None
        self.last_ack_config=None; self.current_run_config=None; self.current_run_id=0; self.last_recorded_step=-1; self.rows=[]; self.latest_revision=0; self.drawn_revision=-1
        self.pending_config=None; self.pending_config_line=""; self.pending_config_token=0; self.pending_config_attempts=0; self.pending_config_auto=False
        self.model_token=0; self.go_attempts=0; self.restore_after_ready=False; self.auto_reconnect_port=None; self.auto_reconnect_attempts=0; self.last_completion_summary=""
        self.latest={"time":0.0,"theta":0.0,"theta_dot":0.0,"alpha":math.pi,"alpha_dot":0.0,"pwm":0.0,"mode":0,"blend":0.0,"pot":0,"enc":0,"action":0.0,"raw":0.0}
        self.settings_path=Path.home()/".rip_ppo_hardware_student.json"
        self.build_ui(); self.build_3d(); self.load_settings(); self.connect_dirty_signals(); self.refresh_ports()
        self.queue_timer=QtCore.QTimer(self); self.queue_timer.timeout.connect(self.poll_queues); self.queue_timer.start(10)
        self.display_timer=QtCore.QTimer(self); self.display_timer.timeout.connect(self.update_display); self.display_timer.start(40)
        self.handshake_timer=QtCore.QTimer(self); self.handshake_timer.timeout.connect(self.handshake_once)
        self.config_retry_timer=QtCore.QTimer(self); self.config_retry_timer.timeout.connect(self.retry_config)
        self.go_retry_timer=QtCore.QTimer(self); self.go_retry_timer.timeout.connect(self.retry_go); self.update_buttons()

    def dspin(self,value,minimum,maximum,decimals,step):
        w=QtWidgets.QDoubleSpinBox(); w.setRange(minimum,maximum); w.setDecimals(decimals); w.setSingleStep(step); w.setValue(value); w.setKeyboardTracking(False); w.setAlignment(QtCore.Qt.AlignRight); return w
    def ispin(self,value,minimum,maximum,step=1):
        w=QtWidgets.QSpinBox(); w.setRange(minimum,maximum); w.setSingleStep(step); w.setValue(value); w.setKeyboardTracking(False); w.setAlignment(QtCore.Qt.AlignRight); return w

    def build_ui(self):
        central=QtWidgets.QWidget(); self.setCentralWidget(central); root=QtWidgets.QHBoxLayout(central)
        scroll=QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff); scroll.setMinimumWidth(440); scroll.setMaximumWidth(525)
        panel=QtWidgets.QWidget(); left=QtWidgets.QVBoxLayout(panel); scroll.setWidget(panel); root.addWidget(scroll)
        connection=QtWidgets.QGroupBox("Serial Connection"); grid=QtWidgets.QGridLayout(connection); self.port_combo=QtWidgets.QComboBox(); self.refresh_button=QtWidgets.QPushButton("Refresh"); self.connect_button=QtWidgets.QPushButton("Connect")
        self.refresh_button.clicked.connect(self.refresh_ports); self.connect_button.clicked.connect(self.toggle_connection); grid.addWidget(self.port_combo,0,0,1,2); grid.addWidget(self.refresh_button,1,0); grid.addWidget(self.connect_button,1,1)
        self.connection_label=QtWidgets.QLabel("Disconnected"); grid.addWidget(self.connection_label,2,0,1,2); left.addWidget(connection)
        model_group=QtWidgets.QGroupBox("PPO Model Selection & STM32 Deployment"); mg=QtWidgets.QGridLayout(model_group); self.model_path=QtWidgets.QLineEdit(); self.model_path.setPlaceholderText("Select ppo_model_weights.h, cached .npz, or a run/deploy folder")
        file_btn=QtWidgets.QPushButton("Choose Model File"); dir_btn=QtWidgets.QPushButton("Choose Run Folder"); file_btn.clicked.connect(self.choose_model_file); dir_btn.clicked.connect(self.choose_model_dir)
        mg.addWidget(self.model_path,0,0,1,2); mg.addWidget(file_btn,1,0); mg.addWidget(dir_btn,1,1); self.model_info=QtWidgets.QLabel("No model selected"); self.model_info.setWordWrap(True); mg.addWidget(self.model_info,2,0,1,2)
        self.upload_progress=QtWidgets.QProgressBar(); self.upload_progress.setRange(0,100); self.upload_progress.setValue(0); mg.addWidget(self.upload_progress,3,0,1,2); left.addWidget(model_group)
        controller=QtWidgets.QGroupBox("Energy Swing-up, PPO Switch & Observer"); form=QtWidgets.QFormLayout(controller)
        self.duration=self.dspin(30,0.1,600,3,1); self.pwm_limit=self.dspin(150,1,255,2,1); self.swing_pwm=self.dspin(120,0,255,2,1); self.kick_time=self.dspin(0.10,0,2,3,0.01)
        self.enter_deg=self.dspin(15,1,60,2,1); self.exit_deg=self.dspin(25,2,90,2,1); self.blend_alpha=self.dspin(0.18,0.001,1,4,0.01); self.velocity_lpf=self.dspin(0.25,0.001,1,4,0.01)
        form.addRow("Duration / s:",self.duration); form.addRow("Safety PWM max:",self.pwm_limit); form.addRow("Energy swing PWM:",self.swing_pwm); form.addRow("Initial kick / s:",self.kick_time)
        form.addRow("PPO enter / deg:",self.enter_deg); form.addRow("PPO exit / deg:",self.exit_deg); form.addRow("Blend λ:",self.blend_alpha); form.addRow("Large-angle velocity LPF β:",self.velocity_lpf)
        note=QtWidgets.QLabel("Far from upright: MPC-firmware phase/energy pumping. Near upright: fixed 5 ms Luenberger observer and the uploaded compact7 7→64→64→1 PPO actor. Hysteresis plus blending prevents abrupt switching.")
        note.setWordWrap(True); note.setStyleSheet("QLabel {color:#333;background:#f2f2f2;padding:6px;}"); form.addRow(note); left.addWidget(controller)
        calibration=QtWidgets.QGroupBox("Sensor Calibration"); cf=QtWidgets.QFormLayout(calibration); self.raw_pot_label=QtWidgets.QLabel("--"); self.raw_enc_label=QtWidgets.QLabel("--")
        self.pot_up=self.ispin(0,0,4095); self.pot_down=self.ispin(0,0,4095); self.theta_scale=self.dspin(0.00583730846,-1,1,11,0.0001); self.motor_sign=QtWidgets.QComboBox(); self.motor_sign.addItem("+1",1); self.motor_sign.addItem("-1",-1)
        self.cal_up_button=QtWidgets.QPushButton("Hold upright → Calibrate α=0"); self.cal_down_button=QtWidgets.QPushButton("Hang down → Calibrate α=π")
        self.cal_up_button.clicked.connect(lambda:self.send_line("CALUP")); self.cal_down_button.clicked.connect(lambda:self.send_line("CALDOWN"))
        cf.addRow("Current pot raw:",self.raw_pot_label); cf.addRow("Current encoder:",self.raw_enc_label); cf.addRow(self.cal_up_button); cf.addRow(self.cal_down_button)
        cf.addRow("PREF / upright raw:",self.pot_up); cf.addRow("PDOWN / hanging raw:",self.pot_down); cf.addRow("θ rad / encoder count:",self.theta_scale); cf.addRow("Motor sign:",self.motor_sign); left.addWidget(calibration)
        actions=QtWidgets.QGroupBox("Run"); av=QtWidgets.QVBoxLayout(actions); self.save_button=QtWidgets.QPushButton("SAVE / Upload PPO Model & Parameters"); self.go_button=QtWidgets.QPushButton("GO"); self.stop_button=QtWidgets.QPushButton("STOP")
        self.save_button.clicked.connect(self.save_and_send); self.go_button.clicked.connect(self.start_run); self.stop_button.clicked.connect(self.stop_run); av.addWidget(self.save_button)
        rr=QtWidgets.QHBoxLayout(); rr.addWidget(self.go_button); rr.addWidget(self.stop_button); av.addLayout(rr); self.status_label=QtWidgets.QLabel("Connect, select PPO model, calibrate, then SAVE. Requires stable firmware protocol v11."); self.status_label.setWordWrap(True); self.state_label=QtWidgets.QLabel(); av.addWidget(self.status_label); av.addWidget(self.state_label); left.addWidget(actions)
        output=QtWidgets.QGroupBox("Result & Logging"); ov=QtWidgets.QVBoxLayout(output); self.csv_checkbox=QtWidgets.QCheckBox("Generate CSV log"); self.csv_checkbox.setChecked(True); self.output_dir=QtWidgets.QLineEdit(os.path.expanduser("~/rip_twin_logs"))
        browse=QtWidgets.QPushButton("Browse"); browse.clicked.connect(self.choose_output_dir); show=QtWidgets.QPushButton("Show Last Result Curves"); show.clicked.connect(self.show_result_dialog); ov.addWidget(self.csv_checkbox)
        orow=QtWidgets.QHBoxLayout(); orow.addWidget(self.output_dir,1); orow.addWidget(browse); ov.addLayout(orow); ov.addWidget(show); left.addWidget(output); left.addStretch(1)
        self.right=QtWidgets.QWidget(); root.addWidget(self.right,1)

    def build_3d(self):
        layout=QtWidgets.QVBoxLayout(self.right); self.figure=Figure(figsize=(9,7),tight_layout=True); self.canvas=FigureCanvas(self.figure); layout.addWidget(self.canvas)
        self.axis=self.figure.add_subplot(111,projection="3d"); self.axis.set_title("Real-time Hardware View",pad=2); self.axis.set_xlabel("X / m"); self.axis.set_ylabel("Y / m"); self.axis.set_zlabel("Z / m")
        limit=ARM_LENGTH+PEND_LENGTH+0.05; self.axis.set_xlim(-limit,limit); self.axis.set_ylim(-limit,limit); self.axis.set_zlim(-0.28,0.38); self.axis.view_init(elev=24,azim=-55)
        angle=np.linspace(0,2*math.pi,80); self.axis.plot(MOTOR_RADIUS*np.cos(angle),MOTOR_RADIUS*np.sin(angle),MOTOR_HEIGHT*np.ones_like(angle),linewidth=2)
        self.axis.plot(MOTOR_RADIUS*np.cos(angle),MOTOR_RADIUS*np.sin(angle),np.zeros_like(angle),linewidth=1.4); self.axis.plot(ARM_LENGTH*np.cos(angle),ARM_LENGTH*np.sin(angle),ARM_Z*np.ones_like(angle),linestyle="--",linewidth=1)
        self.arm_line,=self.axis.plot([],[],[],linewidth=6); self.pendulum_line,=self.axis.plot([],[],[],linewidth=5); self.joint_dot,=self.axis.plot([],[],[],marker="o",markersize=8); self.tip_dot,=self.axis.plot([],[],[],marker="o",markersize=10)
        self.tangent_line,=self.axis.plot([],[],[],linestyle=":",linewidth=2); self.reference_line,=self.axis.plot([],[],[],linestyle="--",linewidth=2.5); self.state_text=self.axis.text2D(0.03,0.87,"",transform=self.axis.transAxes,fontsize=11)

    def numeric_widgets(self): return [self.duration,self.pwm_limit,self.swing_pwm,self.kick_time,self.enter_deg,self.exit_deg,self.blend_alpha,self.velocity_lpf,self.pot_up,self.pot_down,self.theta_scale]
    def connect_dirty_signals(self):
        for w in self.numeric_widgets(): w.valueChanged.connect(self.mark_dirty)
        self.motor_sign.currentIndexChanged.connect(self.mark_dirty); self.model_path.textChanged.connect(self.mark_dirty)
    def mark_dirty(self,*_):
        self.settings_dirty=True; self.config_acknowledged=False; self.model_acknowledged=False; self.save_button.setText("SAVE / Upload PPO Model & Parameters *"); self.update_buttons()

    def choose_model_file(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Select PPO model",str(Path(self.model_path.text() or Path.home()).expanduser()),"PPO model (*.h *.npz)")
        if path:self.model_path.setText(path); self.preview_model()
    def choose_model_dir(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,"Select PPO run folder",str(Path(self.model_path.text() or Path.home()).expanduser()))
        if path:self.model_path.setText(path); self.preview_model()
    def preview_model(self):
        try:
            m=load_model(self.model_path.text().strip())
            self.model_info.setText(f"Resolved: {m.source}\n{m.architecture} | internal normalization | model PWM={m.model_pwm_scale:g}\nID {m.digest} | float32 blob={len(m.float_blob())} bytes")
        except Exception as exc:self.model_info.setText(f"Model not ready: {exc}")

    def refresh_ports(self,preferred=None):
        current=preferred or self.port_combo.currentData(); self.port_combo.clear()
        for port in list(list_ports.comports()): self.port_combo.addItem(f"{port.device} | {port.description}",port.device)
        if current:
            i=self.port_combo.findData(current)
            if i>=0:self.port_combo.setCurrentIndex(i)
    def toggle_connection(self):
        if self.connected:self.disconnect_serial(True)
        else:
            port=self.port_combo.currentData()
            if not port:QtWidgets.QMessageBox.warning(self,"No port","No serial port is selected."); return
            self.open_serial(str(port),False)
    def clear_queues(self):
        for q in (self.line_queue,self.error_queue,self.model_reply_queue):
            while True:
                try:q.get_nowait()
                except queue.Empty:break
    def open_serial(self,port,automatic):
        self.disconnect_serial(False,False); self.clear_queues()
        try:
            try:
                sp=serial.Serial(port,BAUD,timeout=0.05,write_timeout=1.5,exclusive=True)
            except (TypeError, ValueError, NotImplementedError):
                sp=serial.Serial(port,BAUD,timeout=0.05,write_timeout=1.5)
            sp.reset_input_buffer(); sp.reset_output_buffer()
        except Exception as exc:
            if not automatic:QtWidgets.QMessageBox.critical(self,"Serial error",str(exc))
            return False
        self.connection_generation += 1
        generation=self.connection_generation
        self.serial=sp; self.serial_reader=SerialReadThread(sp,self.line_queue,self.error_queue,self.model_reply_queue,generation); self.serial_reader.start(); self.connected=True; self.firmware_ready=False; self.active_port=port
        self.config_acknowledged=False; self.model_acknowledged=False; self.connection_label.setText(f"Opening: {port} @ {BAUD} | protocol v{PROTOCOL_VERSION}"); self.connect_button.setText("Disconnect"); self.handshake_timer.start(350); QtCore.QTimer.singleShot(700,self.handshake_once); self.update_buttons(); return True
    def disconnect_serial(self,send_stop,update_status=True):
        self.handshake_timer.stop(); self.config_retry_timer.stop(); self.go_retry_timer.stop()
        if self.upload_thread is not None and self.upload_thread.isRunning():
            self.upload_thread.cancel()
        old_reader=self.serial_reader
        old_serial=self.serial
        if old_serial is not None and send_stop:
            try:
                self._write_all(b"STOP\n", serial_obj=old_serial)
            except Exception:
                pass
        if old_reader is not None:
            old_reader.stop()
        with self.serial_lock:
            if old_serial is not None:
                try: old_serial.cancel_read()
                except Exception: pass
                try: old_serial.cancel_write()
                except Exception: pass
                try: old_serial.close()
                except Exception: pass
        if old_reader is not None:
            old_reader.join(timeout=1.0)
        self.connection_generation += 1
        self.serial=None; self.serial_reader=None; self.connected=False; self.firmware_ready=False; self.model_acknowledged=False; self.config_acknowledged=False; self.pending_config=None; self.run_pending=False
        self.connection_label.setText("Disconnected"); self.connect_button.setText("Connect")
        if update_status and not self.finishing:self.status_label.setText("Disconnected. Reconnect manually; automatic reconnect is intentionally disabled during deployment.")
        self.update_buttons()
    def _write_all(self, data: bytes, *, serial_obj=None) -> None:
        port = serial_obj if serial_obj is not None else self.serial
        if port is None:
            raise ConnectionError("Serial port is closed")
        view=memoryview(data); offset=0
        with self.serial_lock:
            while offset < len(view):
                written=port.write(view[offset:])
                if written is None or written <= 0:
                    raise serial.SerialTimeoutException("zero-byte serial write")
                offset += int(written)
            port.flush()

    def send_line(self,line,generation=None):
        if not self.connected or self.serial is None:return False
        if generation is not None and int(generation) != self.connection_generation:return False
        try:
            self._write_all((line.strip()+"\n").encode("ascii"))
            return True
        except Exception as exc:
            self.error_queue.put((self.connection_generation,str(exc))); return False
    def send_bytes(self, data: bytes) -> bool:
        if not self.connected or self.serial is None:
            return False
        try:
            view = memoryview(data)
            offset = 0
            with self.serial_lock:
                while offset < len(view):
                    written = self.serial.write(view[offset:])
                    if written is None or written <= 0:
                        raise serial.SerialTimeoutException("zero-byte serial write")
                    offset += int(written)
            return True
        except Exception as exc:
            self.error_queue.put((self.connection_generation,str(exc)))
            return False

    def flush_serial(self) -> bool:
        if not self.connected or self.serial is None:
            return False
        try:
            with self.serial_lock:
                self.serial.flush()
            return True
        except Exception as exc:
            self.error_queue.put((self.connection_generation,str(exc)))
            return False
    def handshake_once(self):
        if self.connected and not self.firmware_ready:self.send_line("HELLO")
    def poll_queues(self):
        try:error_item=self.error_queue.get_nowait()
        except queue.Empty:error_item=None
        if error_item is not None:
            generation,error=error_item
            if generation == self.connection_generation and self.connected:
                was_upload=self.upload_in_progress
                if self.upload_thread is not None and self.upload_thread.isRunning(): self.upload_thread.cancel()
                self.disconnect_serial(False,False)
                self.upload_in_progress=False
                self.status_label.setText(("Serial link lost during model upload; upload stopped cleanly. " if was_upload else "Serial link lost. ") + str(error) + "\nReconnect manually; the panel will not repeatedly reopen the port.")
                if self.recording or self.run_pending:self.recording=False; self.run_pending=False; self.finalize_result()
                self.update_buttons()
                return
        deadline=time.perf_counter()+0.006; count=0
        while count<2000 and time.perf_counter()<deadline:
            try:generation,line=self.line_queue.get_nowait()
            except queue.Empty:break
            if generation != self.connection_generation:
                continue
            self.handle_line(line); count+=1

    def handle_line(self,line):
        if line.startswith("ACK,MODEL") or line.startswith("ERR,MODEL"):
            self.model_reply_queue.put((self.connection_generation,line)); return
        if line.startswith("READY,"):
            parts=line.split(",")
            if len(parts)<3 or parts[2]!=str(PROTOCOL_VERSION):self.status_label.setText(f"Firmware mismatch. Flash the bundled rip_ppo_hardware.ino protocol v{PROTOCOL_VERSION}."); return
            self.firmware_ready=True; self.handshake_timer.stop(); self.connection_label.setText(f"Connected: {self.active_port} @ {BAUD} | stable protocol v{PROTOCOL_VERSION}"); self.send_line("STATUS")
            self.restore_after_ready=False
            self.status_label.setText("Stable serial v11 ready. Select model, calibrate and SAVE. Automatic re-upload is disabled.")
            self.update_buttons(); return
        if line.startswith("ACK,CONFIG,"):
            parts=line.split(",")
            if len(parts)!=3:return
            try:token=int(parts[2])
            except ValueError:return
            if self.pending_config is None or token!=self.pending_config_token:return
            cfg=self.pending_config; automatic=self.pending_config_auto; self.config_retry_timer.stop(); self.pending_config=None; self.pending_config_line=""; self.pending_config_attempts=0; self.pending_config_auto=False
            self.settings_dirty=False; self.config_acknowledged=True; self.last_ack_config=cfg; self.save_button.setText("SAVE / Upload PPO Model & Parameters"); self.save_local_settings()
            self.status_label.setText("Energy swing-up, PPO model and observer parameters are active on STM32." if not automatic else "Serial reconnected; model and parameters restored automatically.")
            self.restore_after_ready=False; self.update_buttons(); return
        if line.startswith("CAL,"):
            parts=line.split(",")
            if len(parts)==3:
                try:value=int(parts[2])
                except ValueError:return
                if parts[1]=="UP":self.pot_up.setValue(value); self.status_label.setText(f"Upright potentiometer zero captured: {value}")
                elif parts[1]=="DOWN":self.pot_down.setValue(value); self.status_label.setText(f"Hanging-down reference captured: {value}")
            return
        if line.startswith("ARMED,"):
            try:run_id=int(line.split(",")[1])
            except Exception:return
            if run_id!=self.current_run_id:return
            self.go_retry_timer.stop(); self.rows.clear(); self.last_recorded_step=-1; self.recording=True; self.run_pending=False; self.latest["time"]=0; self.status_label.setText("Hybrid energy/PPO firmware running at 200 Hz."); self.update_buttons(); return
        if line.startswith("DONE,") or line.startswith("STOPPED,"):
            try:run_id=int(line.split(",")[1])
            except Exception:return
            if run_id==self.current_run_id:self.complete_run("Configured PPO experiment duration completed." if line.startswith("DONE,") else "Run stopped.")
            return
        if line.startswith("ERR,"):
            self.status_label.setText(line); self.update_buttons(); return
        if line.startswith("MON,"):self.handle_monitor(line); return
        if line.startswith("TEL,"):self.handle_telemetry(line)

    def handle_monitor(self,line):
        p=line.split(",")
        if len(p)<7:return
        try:self.latest.update(theta=float(p[1]),theta_dot=float(p[2]),alpha=float(p[3]),alpha_dot=float(p[4]),pwm=0,mode=0,blend=0,pot=int(p[5]),enc=int(p[6]))
        except ValueError:return
        self.latest_revision+=1
    def handle_telemetry(self,line):
        p=line.split(",")
        if len(p)<12:return
        try:
            run_id=int(p[1]); step=int(p[2]); v={"time":step*CONTROL_DT,"theta":float(p[3]),"theta_dot":float(p[4]),"alpha":float(p[5]),"alpha_dot":float(p[6]),"pwm":float(p[7]),"mode":int(p[8]),"blend":float(p[9]),"pot":int(p[10]),"enc":int(p[11]),"action":float(p[12]) if len(p)>12 else 0.0,"raw":float(p[13]) if len(p)>13 else 0.0}
        except ValueError:return
        self.latest.update(v); self.latest_revision+=1
        if self.recording and run_id==self.current_run_id and step>self.last_recorded_step:
            self.rows.append([v["time"],v["theta"],v["theta_dot"],v["alpha"],v["alpha_dot"],v["pwm"],v["mode"],v["blend"],v["action"],v["raw"]]); self.last_recorded_step=step

    def read_config(self):
        cfg=PanelConfig(float(self.duration.value()),float(self.pwm_limit.value()),float(self.swing_pwm.value()),float(self.kick_time.value()),float(self.enter_deg.value()),float(self.exit_deg.value()),float(self.blend_alpha.value()),float(self.velocity_lpf.value()),int(self.pot_up.value()),int(self.pot_down.value()),float(self.theta_scale.value()),int(self.motor_sign.currentData()))
        if cfg.ppo_exit_deg<=cfg.ppo_enter_deg:raise ValueError("PPO exit angle must exceed enter angle.")
        if cfg.swing_pwm>cfg.safety_pwm_limit:raise ValueError("Energy swing PWM must not exceed safety PWM max.")
        if cfg.pot_up==cfg.pot_down:raise ValueError("Upright and hanging-down potentiometer values must differ.")
        if not 0<cfg.velocity_lpf<=1:raise ValueError("Velocity LPF must be in (0,1].")
        return cfg
    def config_line(self,cfg,token):
        return ",".join(["CONFIG",str(PROTOCOL_VERSION),str(token),f"{cfg.duration:.9g}",f"{cfg.safety_pwm_limit:.9g}",f"{cfg.swing_pwm:.9g}",f"{cfg.kick_time:.9g}",f"{cfg.ppo_enter_deg:.9g}",f"{cfg.ppo_exit_deg:.9g}",f"{cfg.blend_alpha:.9g}",f"{cfg.velocity_lpf:.9g}",str(cfg.pot_up),str(cfg.pot_down),f"{cfg.theta_rad_per_count:.12g}",str(cfg.motor_sign)])

    def save_and_send(self):
        if not self.connected or not self.firmware_ready:QtWidgets.QMessageBox.warning(self,"Not ready","Connect STM32 and wait for READY."); return
        try:
            cfg=self.read_config(); source=self.model_path.text().strip()
            if not source:raise ValueError("Select ppo_model_weights.h, a compatible .npz, or a run/deploy directory.")
            model=load_model(source)
        except Exception as exc:QtWidgets.QMessageBox.critical(self,"Model/settings error",str(exc)); return
        self.loaded_model=model; self.last_model_source=str(model.source); self.model_path.setText(str(model.source)); self.model_info.setText(f"{model.architecture} | float32 upload | ID {model.digest}\ninternal normalization; blob={len(model.float_blob())} bytes")
        self.start_model_upload(str(model.source),cfg,automatic=False,preloaded=model)

    def start_model_upload(self,source,cfg,automatic=False,preloaded=None):
        if self.upload_in_progress:return
        try:model=preloaded or load_model(source)
        except Exception as exc:self.status_label.setText(f"Model load failed: {exc}"); return
        self.model_token=(self.model_token+1)%2_000_000_000 or 1; self.upload_in_progress=True; self.model_acknowledged=False; self.config_acknowledged=False; self.upload_progress.setValue(0)
        self.status_label.setText("Starting CRC-verified ASCII float32 PPO upload to STM32 RAM..."); self.upload_thread=ModelUploadThread(self,model,self.model_token,cfg); self.upload_thread.progress.connect(self.on_upload_progress)
        self.upload_thread.completed.connect(lambda ok,msg,obj:self.on_upload_complete(ok,msg,obj,cfg,automatic)); self.upload_thread.start(); self.update_buttons()
    def on_upload_progress(self,value,text):self.upload_progress.setValue(value); self.status_label.setText(text)
    def on_upload_complete(self,ok,message,model,cfg,automatic):
        self.upload_in_progress=False
        self.upload_thread=None
        if not ok:self.status_label.setText(f"Model upload failed: {message}"); self.model_acknowledged=False; self.update_buttons(); return
        if not self.connected or not self.firmware_ready:self.status_label.setText("Model bytes were sent, but the serial connection is no longer active. Reconnect and SAVE again."); self.model_acknowledged=False; self.update_buttons(); return
        self.model_acknowledged=True; self.loaded_model=model; self.last_model_source=str(model.source); self.upload_progress.setValue(100); self.status_label.setText(message+"; sending configuration..."); self.begin_config_send(cfg,automatic)
    def begin_config_send(self,cfg,automatic):
        self.pending_config_token=(self.pending_config_token+1)%2_000_000_000 or 1; self.pending_config=cfg; self.pending_config_line=self.config_line(cfg,self.pending_config_token); self.pending_config_attempts=0; self.pending_config_auto=automatic; self.retry_config(); self.config_retry_timer.start(300); self.update_buttons()
    def retry_config(self):
        if self.pending_config is None:self.config_retry_timer.stop(); return
        if not self.connected or not self.firmware_ready:return
        if self.pending_config_attempts>=12:self.config_retry_timer.stop(); self.status_label.setText("CONFIG failed after retries."); self.pending_config=None; self.update_buttons(); return
        self.pending_config_attempts+=1; self.send_line(self.pending_config_line)

    def start_run(self):
        if not (self.connected and self.firmware_ready and self.model_acknowledged and self.config_acknowledged) or self.settings_dirty:return
        try:self.current_run_config=self.read_config()
        except ValueError as exc:QtWidgets.QMessageBox.warning(self,"Invalid setting",str(exc)); return
        self.current_run_id=int(time.monotonic_ns()//1_000_000)%2_000_000_000 or 1; self.rows.clear(); self.result=None; self.recording=False; self.run_pending=True; self.last_recorded_step=-1; self.go_attempts=0; self.retry_go(); self.go_retry_timer.start(300); self.status_label.setText("Starting energy swing-up with PPO upright hand-off..."); self.update_buttons()
    def retry_go(self):
        if not self.run_pending:self.go_retry_timer.stop(); return
        if self.go_attempts>=8:self.go_retry_timer.stop(); self.run_pending=False; self.status_label.setText("GO not acknowledged."); self.update_buttons(); return
        self.go_attempts+=1; self.send_line(f"GO,{self.current_run_id}")
    def stop_run(self):
        if self.connected:
            self.send_line(f"STOP,{self.current_run_id}"); QtCore.QTimer.singleShot(80,lambda:self.send_line(f"STOP,{self.current_run_id}")); QtCore.QTimer.singleShot(180,lambda:self.send_line(f"STOP,{self.current_run_id}"))
    def complete_run(self,message):
        self.go_retry_timer.stop(); self.recording=False; self.run_pending=False; self.last_completion_summary=self.finalize_result(); self.status_label.setText(message+("\n"+self.last_completion_summary if self.last_completion_summary else "")); self.update_buttons()

    def schedule_auto_reconnect(self,port):
        self.auto_reconnect_port=port
        self.restore_after_ready=False
    def attempt_auto_reconnect(self):
        return

    def update_buttons(self):
        busy=self.recording or self.run_pending or self.upload_in_progress
        self.save_button.setEnabled(self.connected and self.firmware_ready and not busy); self.go_button.setEnabled(self.connected and self.firmware_ready and self.model_acknowledged and self.config_acknowledged and not self.settings_dirty and not busy)
        self.stop_button.setEnabled(self.connected and (self.recording or self.run_pending)); self.cal_up_button.setEnabled(self.connected and self.firmware_ready and not busy); self.cal_down_button.setEnabled(self.connected and self.firmware_ready and not busy)
        self.connect_button.setEnabled(not self.upload_in_progress)
    def update_display(self):
        if self.latest_revision==self.drawn_revision:return
        self.drawn_revision=self.latest_revision; v=self.latest; center,joint,tip,ref,tangent=rip_points(v["theta"],v["alpha"]); set_line3d(self.arm_line,center,joint); set_line3d(self.pendulum_line,joint,tip); set_point3d(self.joint_dot,joint); set_point3d(self.tip_dot,tip); set_line3d(self.reference_line,joint,ref); set_line3d(self.tangent_line,joint,joint+0.11*tangent)
        mode=MODE_NAMES.get(v["mode"],str(v["mode"])); self.state_text.set_text(f"t = {v['time']:7.3f} s\nθ = {v['theta']: .4f} rad\nα = {v['alpha']: .4f} rad\nPWM = {v['pwm']: .0f}\nmode = {mode}\nobserver blend = {v['blend']:.3f}\naction = {v['action']:+.3f}  raw={v['raw']:+.3f}")
        self.state_label.setText(f"θ={v['theta']:+.4f}, θ̇={v['theta_dot']:+.4f}, α={v['alpha']:+.4f}, α̇={v['alpha_dot']:+.4f}, PWM={v['pwm']:+.0f}, {mode}")
        self.raw_pot_label.setText(str(v["pot"])); self.raw_enc_label.setText(str(v["enc"])); self.canvas.draw_idle()
    def rows_to_result(self):
        if not self.rows:return None
        d=np.asarray(self.rows,float); return HardwareResult(d[:,0],d[:,1],d[:,2],d[:,3],d[:,4],d[:,5],d[:,6].astype(int),d[:,7],d[:,8],d[:,9])
    def finalize_result(self):
        self.result=self.rows_to_result()
        if self.result is None:return "No telemetry samples were recorded."
        cfg=self.current_run_config or self.last_ack_config or PanelConfig(); png,csvp=default_output_paths(self.output_dir.text(),cfg.duration); fig=build_result_figure(self.result); fig.savefig(png,dpi=300,bbox_inches="tight"); self.last_png_path=png
        if self.csv_checkbox.isChecked():save_result_csv(self.result,csvp); self.last_csv_path=csvp
        metrics=result_metrics(self.result); return f"Saved PNG: {png}"+(f"\nSaved CSV: {csvp}" if self.csv_checkbox.isChecked() else "")+f"\nmean(|alpha|)={metrics['alpha_abs_mean']:.6f} rad, mean(|PWM|)={metrics['pwm_abs_mean']:.3f}"
    def show_result_dialog(self):
        if self.result is None:QtWidgets.QMessageBox.information(self,"No result","No completed hardware result yet."); return
        dialog=QtWidgets.QDialog(self); dialog.setWindowTitle(f"RIP Hardware Hybrid PPO Result | {self.result.time[-1]:.3f} s Response"); dialog.resize(1080,800); layout=QtWidgets.QVBoxLayout(dialog); canvas=FigureCanvas(build_result_figure(self.result)); layout.addWidget(canvas); close=QtWidgets.QPushButton("Close"); close.clicked.connect(dialog.close); layout.addWidget(close); dialog.exec_()
    def choose_output_dir(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,"Output directory",self.output_dir.text())
        if path:self.output_dir.setText(path)
    def save_local_settings(self):
        try:
            data=asdict(self.last_ack_config or self.read_config()); data.update(model_source=self.last_model_source,output_dir=self.output_dir.text(),save_csv=self.csv_checkbox.isChecked())
            self.settings_path.write_text(json.dumps(data,indent=2),encoding="utf-8")
        except Exception:pass
    def load_settings(self):
        if not self.settings_path.exists():return
        try:data=json.loads(self.settings_path.read_text(encoding="utf-8"))
        except Exception:return
        mapping={"duration":self.duration,"safety_pwm_limit":self.pwm_limit,"swing_pwm":self.swing_pwm,"kick_time":self.kick_time,"ppo_enter_deg":self.enter_deg,"ppo_exit_deg":self.exit_deg,"blend_alpha":self.blend_alpha,"velocity_lpf":self.velocity_lpf,"pot_up":self.pot_up,"pot_down":self.pot_down,"theta_rad_per_count":self.theta_scale}
        for key,w in mapping.items():
            if key in data:w.setValue(data[key])
        idx=self.motor_sign.findData(int(data.get("motor_sign",1))); self.motor_sign.setCurrentIndex(max(idx,0)); self.model_path.setText(str(data.get("model_source",""))); self.output_dir.setText(str(data.get("output_dir",os.path.expanduser("~/rip_twin_logs")))); self.csv_checkbox.setChecked(bool(data.get("save_csv",True)))
        if self.model_path.text():self.preview_model()
        self.settings_dirty=True
    def closeEvent(self,event):
        self.finishing=True; self.disconnect_serial(True,False); event.accept()


def main() -> int:
    app=QtWidgets.QApplication(sys.argv); win=MainWindow(); win.show(); return app.exec_()

if __name__=="__main__": raise SystemExit(main())
