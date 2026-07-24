import json
from pathlib import Path

base = Path(r"C:\Users\kreml\AI-Enabled-PKU\lab3\4.TD3\train\runs")
runs = [
    "benchmark_td3_rip_original_2m_then_dr_20260721_113956",
    "td3_rip_original_2m_then_dr_20260722_115841",
    "td3_rip_original_2m_then_dr_20260722_121830",
    "td3_rip_original_2m_then_dr_20260722_200150",
]


def summarize(run: str) -> None:
    d = base / run
    print("=" * 70)
    print(run)
    print("exists", d.exists())
    if not d.exists():
        return

    for rel in [
        "eval_history.json",
        "eval_history.csv",
        "training_summary.json",
        "config_snapshot.json",
        "best_nominal_model/best_metrics.json",
        "best_randomized_model/best_metrics.json",
        "best_model/selection.json",
        "episode_history.csv",
    ]:
        p = d / rel
        print(f"  {rel}:", "YES" if p.exists() else "no", f"({p.stat().st_size} B)" if p.exists() else "")

    for label in ["best_nominal_model", "best_randomized_model"]:
        p = d / label / "best_metrics.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            keys = [
                "step",
                "mean_reward_per_step",
                "stable_success_rate",
                "capture_rate",
                "mean_maintain_max_stable",
                "randomization_level",
                "eval_randomization_level",
                "phase",
                "mean_episode_reward",
            ]
            print(f"  -- {label} --")
            for k in keys:
                if k in m:
                    print(f"     {k}: {m[k]}")

    p = d / "best_model" / "selection.json"
    if p.exists():
        print("  -- selection --", p.read_text(encoding="utf-8")[:800])

    eh = d / "eval_history.json"
    if not eh.exists():
        return
    rows = json.loads(eh.read_text(encoding="utf-8"))
    print(f"  eval entries: {len(rows)}")
    if not rows:
        return

    last = rows[-1]
    print(
        "  last:",
        f"step={last.get('step')}",
        f"phase={last.get('phase')}",
        f"level={last.get('eval_randomization_level', last.get('randomization_level'))}",
        f"reward/step={last.get('mean_reward_per_step')}",
        f"stable={last.get('stable_success_rate')}",
        f"capture={last.get('capture_rate')}",
    )

    nom = [
        r
        for r in rows
        if float(r.get("eval_randomization_level", r.get("randomization_level", 1))) == 0.0
    ]
    if nom:
        best = max(
            nom,
            key=lambda r: (r.get("stable_success_rate", 0), r.get("mean_reward_per_step", -1e9)),
        )
        print(
            "  best nominal eval:",
            f"step={best.get('step')}",
            f"reward/step={best.get('mean_reward_per_step')}",
            f"stable={best.get('stable_success_rate')}",
            f"capture={best.get('capture_rate')}",
            f"maintain={best.get('mean_maintain_max_stable')}",
        )

    by_level: dict[float, list] = {}
    for r in rows:
        lvl = float(r.get("eval_randomization_level", r.get("randomization_level", -1)))
        by_level.setdefault(lvl, []).append(r)
    print("  by eval level (best stable, best reward):")
    for lvl in sorted(by_level):
        rs = by_level[lvl]
        b = max(
            rs,
            key=lambda r: (r.get("stable_success_rate", 0), r.get("mean_reward_per_step", -1e9)),
        )
        print(
            f"    L={lvl}: n={len(rs)}",
            f"best_stable={b.get('stable_success_rate')}",
            f"best_rps={b.get('mean_reward_per_step')}",
            f"@step={b.get('step')}",
            f"capture={b.get('capture_rate')}",
        )


for run in runs:
    summarize(run)
