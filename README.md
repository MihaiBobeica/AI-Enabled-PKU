# AI-Enabled PKU — Rotary Inverted Pendulum (RIP)

Self-contained tools for a Furuta / rotary inverted pendulum: a ROSRIP-aligned digital twin simulator and STM32 hardware control with dual-PID (swing pump → blend → balance).

## Contents

| File | Description |
|------|-------------|
| `rip_dual_pid.py` | Digital twin simulator (GUI or headless), 200 Hz control loop |
| `rip_dual_pid_hardware.py` | PyQt GUI for live STM32 runs over serial |
| `rip_dual_pid_hardware.ino` | STM32 firmware (Arduino framework) |
| `requirements.txt` | Python dependencies |

## Requirements

- **Python** 3.9+ (3.10–3.13 recommended)
- Packages in `requirements.txt`: `numpy`, `matplotlib`, `PyQt5`, `pyserial`
- **Hardware path only:** STM32 board, Arduino IDE / PlatformIO, USB serial driver

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Digital twin (simulator)

GUI:

```powershell
python rip_dual_pid.py
```

Headless (no Qt window; still needs matplotlib for plots):

```powershell
python rip_dual_pid.py --headless
```

Useful flags: `--duration`, `--initial` (`downward` / `upright+8` / `upright-8`), `--output-dir`, `--no-csv`, `--no-noise`, PID and swing / blend gains (`--kp-alpha`, …). Logs default to `~/rip_twin_logs`.

## Hardware

1. Flash `rip_dual_pid_hardware.ino` to the STM32 (Arduino core; control loop at 200 Hz).
2. Connect USB; note the COM port.
3. Run the host GUI:

```powershell
python rip_dual_pid_hardware.py
```

4. Select the serial port (baud **921600**), wait for READY, calibrate, **SAVE** parameters to the board, then **GO**.

Pins used by the sketch (STM32): pot `A0`, encoder `2`/`3`, motor `IN1=4`, `IN2=5`, `STBY=7`, `PWM=10`.

## Notes

- Simulator timing, PWM convention, and geometry are aligned with the hardware project for comparable runs.
- `pyserial` is only required for `rip_dual_pid_hardware.py`.
- GUI mode for the twin needs `PyQt5`; use `--headless` if Qt is unavailable.
