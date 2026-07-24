import json
from pathlib import Path

base = Path(r"C:\Users\kreml\AI-Enabled-PKU\lab3\4.TD3\train\runs")
ta_name = "benchmark_td3_rip_original_2m_then_dr_20260721_113956"
you_name = "td3_rip_original_2m_then_dr_20260722_200150"


def load_eval(run: str):
    return json.loads((base / run / "eval_history.json").read_text(encoding="utf-8"))


def level_of(r):
    return float(r.get("eval_randomization_level", r.get("randomization_level", -1)))


def summarize(run: str):
    rows = load_eval(run)
    print("=" * 70)
    print(run)
    print("keys sample:", sorted(rows[0].keys()))
    print("n_evals", len(rows), "last_step", rows[-1].get("step"), "last_phase", rows[-1].get("phase"))
    for lvl in sorted({level_of(r) for r in rows}):
        rs = [r for r in rows if level_of(r) == lvl]
        full_stable = sum(1 for r in rs if float(r.get("stable_success_rate", 0)) >= 0.999)
        full_capture = sum(1 for r in rs if float(r.get("capture_rate", 0)) >= 0.999)
        last = rs[-1]
        print(
            f"  L={lvl}: evals={len(rs)} full_stable={full_stable} full_capture={full_capture}"
            f" last_step={last.get('step')} last_stable={last.get('stable_success_rate')}"
            f" last_capture={last.get('capture_rate')}"
        )
        # best by stable then control_score / mean_reward if present
        def score(r):
            return (
                float(r.get("stable_success_rate", 0)),
                float(r.get("capture_rate", 0)),
                float(r.get("control_score") or r.get("mean_reward") or r.get("mean_reward_per_step") or -1e18),
            )

        best = max(rs, key=score)
        print(
            "    best:",
            f"step={best.get('step')}",
            f"stable={best.get('stable_success_rate')}",
            f"capture={best.get('capture_rate')}",
            f"mean_reward={best.get('mean_reward')}",
            f"control_score={best.get('control_score')}",
            f"mean_abs_alpha={best.get('mean_abs_alpha')}",
            f"mean_abs_pwm={best.get('mean_abs_pwm')}",
            f"mean_length={best.get('mean_length')}",
        )


for name in [ta_name, you_name]:
    summarize(name)

# best metrics files if present
print("=" * 70)
print("BEST METRICS FILES")
for name in [ta_name, you_name]:
    for label in ["best_nominal_model", "best_randomized_model", "best_model"]:
        p = base / name / label / "best_metrics.json"
        if p.exists():
            print(f"-- {name} / {label} --")
            print(p.read_text(encoding="utf-8"))

# config diffs
print("=" * 70)
print("CONFIG DIFFS (leaf)")


def dig(d, path=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                out.update(dig(v, p))
            else:
                out[p] = v
    return out


ta = dig(json.loads((base / ta_name / "config_snapshot.json").read_text(encoding="utf-8")))
you = dig(json.loads((base / you_name / "config_snapshot.json").read_text(encoding="utf-8")))
common = set(ta) & set(you)
skip = ("path", "dir", "run", "seed_time", "created", "host")
diffs = []
for k in sorted(common):
    if any(s in k.lower() for s in skip):
        continue
    if ta[k] != you[k]:
        diffs.append((k, ta[k], you[k]))
print("diff_count", len(diffs))
for k, a, b in diffs:
    print(f"{k}:\n  TA = {a!r}\n  YOU= {b!r}")

# first time each run hit stable 1.0 at L=0 and L=0.5
print("=" * 70)
print("FIRST SUCCESS MILESTONES")
for name in [ta_name, you_name]:
    rows = load_eval(name)
    for lvl in [0.0, 0.5]:
        hit = next(
            (
                r
                for r in rows
                if level_of(r) == lvl and float(r.get("stable_success_rate", 0)) >= 0.999
            ),
            None,
        )
        print(f"{name} L={lvl}: first full stable @ step={None if not hit else hit.get('step')}")
