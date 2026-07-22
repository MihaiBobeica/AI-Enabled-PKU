#!/usr/bin/env python3
"""Validate/adapt a TD3 training run to canonical deployment NPZ.

Normal deployment does not require this script: the hardware and simulation
panels now prefer an existing *.td3_deploy.npz automatically.  This utility is
provided as a one-click fallback and writes deploy/td3_actor.td3_deploy.npz.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from rip_td3_sim_test import load_model


def choose_dir() -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.update()
    path = filedialog.askdirectory(title="Select TD3 training run directory")
    root.destroy()
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--no-gui", action="store_true")
    args = ap.parse_args()
    selected = args.run_dir or ("" if args.no_gui else choose_dir())
    if not selected:
        print("No directory selected.")
        return 2
    run = Path(selected).expanduser().resolve()
    model = load_model(run)
    out = run / "deploy" / "td3_actor.td3_deploy.npz"
    model.save_npz(out)
    check = load_model(out)
    if check.digest != model.digest:
        raise RuntimeError("Converted model digest mismatch")
    print(f"OK: {model.architecture}")
    print(f"source: {model.source}")
    print(f"output: {out}")
    print(f"digest: {model.digest}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
