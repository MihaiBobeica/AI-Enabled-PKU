# Lab 3 — MPC, DQN, PPO, TD3

```text
lab3/
  1.MPC/            # MPC simulation + hardware + Optuna search
  2.DQN/train/      # DQN training + Optuna search
  2.DQN/deploy/     # DQN sim/hardware deploy
  3.PPO/train/      # PPO training + Optuna search
  3.PPO/deploy/     # PPO sim/hardware deploy
  4.TD3/train/      # TD3 training + Optuna search
  4.TD3/deploy/     # TD3 sim/hardware deploy
  docs/             # manuals + report template
  submission/       # Report-3 packaging scaffold
```

Original course zip archives remain next to each method folder.

## Guided hyperparameter search (Optuna TPE)

Each method has an independent `search_params.py` with its own `--hours` budget and `--resume`.

```powershell
# Terminal 1 — MPC
cd lab3\1.MPC
python search_params.py --hours 2

# Terminal 2 — DQN
cd lab3\2.DQN\train
python search_params.py --hours 8

# Terminal 3 — PPO
cd lab3\3.PPO\train
python search_params.py --hours 8

# Terminal 4 — TD3
cd lab3\4.TD3\train
python search_params.py --hours 8
```

Resume after interrupt:

```powershell
python search_params.py --hours 8 --resume
```

Artifacts per method folder: `optuna_study.db`, `BEST.json`, `best_run/`, `search_results.csv`, `search_logs/`, `figures/`.

Requires: `pip install optuna`.

After search, apply winners and run full overnight trains:

```powershell
cd lab3\2.DQN\train
python apply_best.py
# merge best_config_overrides.json into config.py
python run.py --worker train
```

See [docs/OVERNIGHT_FULL_TRAIN.md](docs/OVERNIGHT_FULL_TRAIN.md) and [submission/README.md](submission/README.md).

