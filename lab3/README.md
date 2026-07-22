# Lab 3 — MPC, DQN, PPO, TD3

```text
lab3/
  1.MPC/            # MPC simulation + hardware
  2.DQN/train/      # DQN training
  2.DQN/deploy/     # DQN sim/hardware deploy
  3.PPO/train/      # PPO training
  3.PPO/deploy/     # PPO sim/hardware deploy
  4.TD3/train/      # TD3 training
  4.TD3/deploy/     # TD3 sim/hardware deploy
  suite/            # automated PPO/DQN best-suite runner
  docs/             # manuals + report template
```

Original course zip archives remain next to each method folder.

## Best-suite training (PPO + DQN)

```powershell
cd lab3\suite
python run_best_suite.py --hours 6
```

See [suite/SUITE_README.md](suite/SUITE_README.md).
