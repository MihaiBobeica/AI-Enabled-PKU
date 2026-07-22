# Submission scaffold for Lab Report 3

Expected final ZIP layout (see course template):

```text
GroupXX_LabReport3.zip
|-- GroupXX_LabReport3.pdf
|-- configs/
|-- models/
|-- training/
|-- data/   (mpc|dqn|ppo|td3)/(sim|hardware)/
`-- bonus/  (optional)
```

## Workflow

1. Run Optuna searches (`search_params.py --hours …`) in four terminals.
2. `python apply_best.py` in each method folder; merge overrides; full overnight train for DQN/PPO/TD3.
3. Confirm [`docs/common_test_protocol.json`](../docs/common_test_protocol.json).
4. MPC sim 10-run: `python lab3/1.MPC/run_fixed_sim_trials.py`
5. RL/hardware 10-run campaigns via each method's deploy panels (archive CSVs under `submission/data/...`).
6. Fill [`docs/report3.tex`](../docs/report3.tex) / PDF.
7. `python lab3/package_submission.py --group 01`
