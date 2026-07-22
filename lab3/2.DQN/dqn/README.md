# DQN

## Start

```bash
cd dqn
python3 run.py
```

Command-line workers:

```bash
python3 run.py --worker train
python3 run.py --worker smoke
python3 run.py --worker test --run-dir <run_dir> --variant current
```

Training result directories are named:

```text
runs/dqn_YYYYMMDD_HHMMSS/
```

## Training schedule

- 0–2,000,000 steps: nominal, randomization level 0.00
- 2,000,000–3,000,000 steps: level 0.10
- 3,000,000–4,000,000 steps: level 0.30
- 4,000,000–5,000,000 steps: level 0.50

All training, environment, reward and randomization parameters are unchanged from
the supplied package.

## Best-model rule

Within the nominal phase and randomized phases, best models are tracked
separately. A new evaluation replaces the current best when:

1. `stable_success_rate` is higher; or
2. success rate is exactly tied and `mean_reward_per_step` improves by more than
   `EVAL.early_stop_reward_min_delta` (currently 0.02).

A randomized best becomes the default deployment model only when its success
rate reaches `EVAL.randomized_model_min_success_for_selection` (currently 0.60).
Otherwise the nominal best is used.

## Direct deployment

There is no distillation stage. Training directly writes the selected deployment
network and C header:

```text
selected_best_model.pt
deploy/model_weights.h
```

The trained network is already `6 -> 64 -> 64 -> 10`, so no student model is
created.

Specific model checkpoint filenames are unchanged, including:

```text
best_nominal_model/best_model.pt
recovery_model/nominal_2m_last.pt
recovery_model/matlab_success_snapshot.pt
best_randomized_model/best_model.pt
selected_best_model.pt
final_model.pt
deploy/model_weights.h
```

## Local dependencies

The launcher gives priority to the package-local copies of Gymnasium,
cloudpickle and Stable-Baselines3. Torch, NumPy, Matplotlib and Tkinter are
provided by the active Python environment.
