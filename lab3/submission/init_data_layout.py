"""Placeholder helpers documenting how to archive 10 fixed RL sim trials.

After overnight training + deploy model selection, run each method's sim test
panel / CLI 10 times with the common protocol seeds and copy outputs to:

  lab3/submission/data/<method>/sim/<METHOD>_SIM_R01.csv ...

Hardware trials go under:

  lab3/submission/data/<method>/hardware/<METHOD>_HW_R01.csv ...

Bonus-2 PPO single-policy trials go under:

  lab3/submission/bonus/ppo_single_policy/{sim,hardware}/

Keep failed trials. Do not replace failures with extra favourable runs.
"""
from __future__ import annotations

from pathlib import Path

METHODS = ("dqn", "ppo", "td3", "mpc")


def ensure_layout(root: Path) -> None:
    for method in METHODS:
        for split in ("sim", "hardware"):
            (root / "data" / method / split).mkdir(parents=True, exist_ok=True)
            readme = root / "data" / method / split / "README.txt"
            if not readme.exists():
                tag = "SIM" if split == "sim" else "HW"
                readme.write_text(
                    f"Place 10 fixed {method.upper()} {split} trial CSVs/figures here "
                    f"(e.g. {method.upper()}_{tag}_R01.csv).\n",
                    encoding="utf-8",
                )

    for split in ("sim", "hardware"):
        dest = root / "bonus" / "ppo_single_policy" / split
        dest.mkdir(parents=True, exist_ok=True)
        readme = dest / "README.txt"
        if not readme.exists():
            tag = "SIM" if split == "sim" else "HW"
            readme.write_text(
                f"Bonus-2 PPO single-policy {split} trials "
                f"(e.g. PPO_BONUS2_{tag}_R01.csv). No hybrid swing-up.\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    ensure_layout(Path(__file__).resolve().parent)
    print("Created submission data + bonus/ppo_single_policy placeholders")
