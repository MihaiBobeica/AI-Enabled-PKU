# Lab3 One-Command Best Suite

Automated PPO ablations, distillation, and leftover DQN training with a single long-running command.

## Layout

```text
lab3/
  1.MPC/          # MPC sim + hardware
  2.DQN/train/    # DQN training package
  2.DQN/deploy/   # DQN deploy
  3.PPO/train/    # PPO training package
  3.PPO/deploy/   # PPO deploy
  4.TD3/train/    # TD3 training package
  4.TD3/deploy/   # TD3 deploy
  suite/          # this folder
  docs/           # PDFs / notes
```

## Launch (6 hours)

```powershell
cd c:\Users\kreml\AI-Enabled-PKU\lab3\suite
python run_best_suite.py --hours 6
```

Keep the PC awake and plugged in. Do not edit `3.PPO/train/config.py` or `2.DQN/train/config.py` while the suite runs.

## Resume after interrupt

```powershell
cd c:\Users\kreml\AI-Enabled-PKU\lab3\suite
python run_best_suite.py --hours 6 --resume
```

If the saved deadline expired, pass `--hours` again to extend.

## Other modes

| Command | Purpose |
|---------|---------|
| `python run_best_suite.py --dry-run` | Print trial queue, no training |
| `python run_best_suite.py --smoke-only` | Quick PPO+DQN pipeline check (~minutes) |
| `python run_best_suite.py --hours 6 --ppo-only` | Skip DQN |
| `python run_best_suite.py --hours 6 --dqn-only` | Skip PPO ablations |
| `python run_best_suite.py --hours 6 --skip-smoke` | Skip initial smoke tests |
| `python run_best_suite.py --hours 6 --verbose` | Echo full trainer stdout (default is quiet) |

## What it runs

1. Smoke tests (unless skipped)
2. PPO one-factor trials: P0–P4 (longer train, slower DR, entropy, KL, LR)
3. Seed repeats of the best PPO config
4. Distill best teacher → `model_weights.h`
5. Optional headless digital-twin checks
6. DQN benchmark (+ one ablation if time remains)

## Outputs

| Path | Description |
|------|-------------|
| [BEST.md](BEST.md) | Winning PPO/DQN paths and scores |
| `suite_experiment_log.csv` | Every trial (including failures) |
| `suite_state.json` | Resume checkpoint |
| `suite_logs/` | Per-trial subprocess logs |
| `_suite_pristine/` | Frozen config snapshots (safe restore) |
| `../3.PPO/train/runs/suite_*` | PPO training runs |
| `../2.DQN/train/runs/dqn_*` | DQN training runs |

## Best selection

- **PPO:** max randomized eval `score` in `eval_logs/eval_metrics.csv` (baseline ~20741)
- **DQN:** `training_summary.json` / stable success rate; deploy via `selected_best_model.pt`
