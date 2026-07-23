# Overnight full training (after Optuna BEST)

CPU thread caps are set to `torch_num_threads: 2` in DQN/PPO/TD3 configs for coexistence.

## Apply search winners

```powershell
cd lab3\2.DQN\train
python apply_best.py
# merge best_config_overrides.json into config.py, then:
python run.py --worker train

cd lab3\3.PPO\train
python apply_best.py
python run.py train

cd lab3\4.TD3\train
python apply_best.py
python run.py --worker train
```

Full lengths: DQN/TD3 **5,000,000** steps; PPO **2,000,000** steps.

Select **best** checkpoints (not necessarily final). For TD3 use `best_model.zip` via Choose Model File.
