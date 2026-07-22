"""Assemble GroupXX_LabReport3.zip scaffold from available artifacts.

Usage:
  python package_submission.py --group 01
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def copy_if(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="XX")
    parser.add_argument("--lab3", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    lab3: Path = args.lab3
    out = lab3 / "submission" / f"Group{args.group}_LabReport3"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in (f"Group{args.group}_LabReport3.pdf", "report3.pdf", "docs/report3.tex"):
        p = lab3 / name if not str(name).startswith("docs") else lab3 / name
        if p.exists():
            copy_if(p, out / p.name)

    configs = out / "configs"
    models = out / "models"
    training = out / "training"
    data = out / "data"
    bonus = out / "bonus"
    for d in (configs, models, training, data, bonus):
        d.mkdir(parents=True, exist_ok=True)

    mapping = [
        (lab3 / "2.DQN" / "train" / "BEST.json", configs / "dqn_BEST.json"),
        (lab3 / "2.DQN" / "train" / "best_config_overrides.json", configs / "dqn_config_snapshot.json"),
        (lab3 / "3.PPO" / "train" / "BEST.json", configs / "ppo_BEST.json"),
        (lab3 / "3.PPO" / "train" / "best_config_overrides.json", configs / "ppo_config_snapshot.json"),
        (lab3 / "3.PPO" / "train" / "BEST_bonus2.json", configs / "ppo_bonus2_BEST.json"),
        (lab3 / "3.PPO" / "train" / "best_config_overrides_bonus2.json", configs / "ppo_bonus2_config_snapshot.json"),
        (lab3 / "3.PPO" / "train" / "config_bonus2.py", configs / "ppo_bonus2_config.py"),
        (lab3 / "4.TD3" / "train" / "BEST.json", configs / "td3_BEST.json"),
        (lab3 / "4.TD3" / "train" / "best_config_overrides.json", configs / "td3_config_snapshot.json"),
        (lab3 / "1.MPC" / "frozen_mpc_params.json", configs / "mpc_frozen_params.json"),
        (lab3 / "1.MPC" / "BEST.json", configs / "mpc_BEST.json"),
        (lab3 / "docs" / "BONUS_CLAIMS.md", bonus / "BONUS_CLAIMS.md"),
    ]
    for src, dst in mapping:
        copy_if(src, dst)

    copy_if(lab3 / "2.DQN" / "train" / "best_run", models / "dqn_best_run")
    copy_if(lab3 / "3.PPO" / "train" / "best_run", models / "ppo_best_run")
    copy_if(lab3 / "3.PPO" / "train" / "best_run_bonus2", models / "ppo_bonus2_best_run")
    copy_if(lab3 / "4.TD3" / "train" / "best_run", models / "td3_best_run")

    for method, folder in (
        ("dqn", lab3 / "2.DQN" / "train"),
        ("ppo", lab3 / "3.PPO" / "train"),
        ("td3", lab3 / "4.TD3" / "train"),
        ("mpc", lab3 / "1.MPC"),
    ):
        dest = training / method
        dest.mkdir(parents=True, exist_ok=True)
        copy_if(folder / "search_results.csv", dest / "search_results.csv")
        copy_if(folder / "figures", dest / "figures")
        copy_if(folder / "BEST.json", dest / "BEST.json")

    # Bonus 1 evidence: Optuna history / BEST files
    b1 = bonus / "bonus1_evidence"
    b1.mkdir(parents=True, exist_ok=True)
    for method, folder in (
        ("dqn", lab3 / "2.DQN" / "train"),
        ("ppo", lab3 / "3.PPO" / "train"),
        ("td3", lab3 / "4.TD3" / "train"),
    ):
        dest = b1 / method
        dest.mkdir(parents=True, exist_ok=True)
        copy_if(folder / "BEST.json", dest / "BEST.json")
        copy_if(folder / "figures", dest / "figures")
        copy_if(folder / "search_results.csv", dest / "search_results.csv")

    # Bonus 2 evidence
    b2 = bonus / "ppo_single_policy"
    b2.mkdir(parents=True, exist_ok=True)
    ppo_train = lab3 / "3.PPO" / "train"
    copy_if(ppo_train / "BEST_bonus2.json", b2 / "BEST_bonus2.json")
    copy_if(ppo_train / "best_run_bonus2", b2 / "best_run_bonus2")
    copy_if(ppo_train / "search_results_bonus2.csv", b2 / "search_results_bonus2.csv")
    copy_if(ppo_train / "figures_bonus2", b2 / "figures")
    copy_if(ppo_train / "config_bonus2.py", b2 / "config_bonus2.py")
    (b2 / "sim").mkdir(exist_ok=True)
    (b2 / "hardware").mkdir(exist_ok=True)

    src_data = lab3 / "submission" / "data"
    if src_data.exists():
        for item in src_data.iterdir():
            copy_if(item, data / item.name)

    src_bonus_data = lab3 / "submission" / "bonus"
    if src_bonus_data.exists():
        for item in src_bonus_data.iterdir():
            copy_if(item, bonus / item.name)

    copy_if(lab3 / "docs" / "course_benchmark_baseline.json", out / "course_benchmark_baseline.json")
    copy_if(lab3 / "docs" / "common_test_protocol.json", out / "common_test_protocol.json")

    manifest = {
        "group": args.group,
        "bonuses_claimed": ["Bonus1", "Bonus2"],
        "bonus3_claimed": False,
        "note": "Fill GroupXX_LabReport3.pdf, complete 10x10 data folders, then re-run this packager.",
        "layout": ["configs/", "models/", "training/", "data/", "bonus/"],
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = lab3 / "submission" / f"Group{args.group}_LabReport3.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(Path(f"Group{args.group}_LabReport3") / path.relative_to(out)))
    print(f"Wrote {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
