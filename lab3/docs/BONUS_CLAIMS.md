# Lab Report 3 — Bonus claims

## Claimed

| Bonus | Status | Notes |
|-------|--------|-------|
| **Bonus 1** | Claimed | Clear/substantial improvement on the strongest RL method after Optuna + full overnight train (DQN / PPO / TD3). Fill method name + numbers in `report3.tex` §7.2 when evidence exists. |
| **Bonus 2** | Claimed | Separate PPO **single-policy** swing-up+stabilize track (`config_bonus2.py`). No energy/PID/LQR/MPC switching. Use Bonus-2 deploy panels + 10 sim / 10 HW tables in §7.3. |
| **Bonus 3** | **Not claimed** | No ROS unified framework in this submission. |

## Commands (you run these)

### Core Optuna (supports Bonus 1 evidence)

```powershell
cd lab3\1.MPC
python search_params.py --hours 2

cd lab3\2.DQN\train
python search_params.py --hours 8

cd lab3\3.PPO\train
python search_params.py --hours 8

cd lab3\4.TD3\train
python search_params.py --hours 8
```

### Bonus 2 Optuna (isolated artifacts)

```powershell
cd lab3\3.PPO\train
python search_params_bonus2.py --hours 8
python apply_best_bonus2.py
# merge best_config_overrides_bonus2.json into config_bonus2.py PPO dict, then:
python run.py train --config config_bonus2
```

### Distill Bonus-2 teacher → compact7 (after full train)

Prefer loading the SB3 `.zip` directly in Bonus-2 deploy panels (ReLU/ReLU/Tanh).
Optional distill for a C header / lighter upload:

```powershell
cd lab3\3.PPO\train
python distill_student64_deploy.py --config config_bonus2 --activation ReLU --run-dir runs\<ppo_bonus2_run>
```

### Deploy (always-on policy, no hybrid swing-up)

```powershell
cd lab3\3.PPO\deploy
python rip_ppo_bonus2_sim_test.py
python rip_ppo_bonus2_hardware.py
```

Flash firmware from `lab3/3.PPO/deploy/rip_ppo_bonus2_hardware/rip_ppo_bonus2_hardware.ino` (always NN; leave hybrid `rip_ppo_hardware` untouched for Part IV).

Archive Bonus-2 trials under `lab3/submission/bonus/ppo_single_policy/{sim,hardware}/`.

## Evidence packing

`python lab3/package_submission.py --group XX` copies Optuna figures/`BEST*.json` into `training/` and `bonus/`.
