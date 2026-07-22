# PPO Stage-1 Design Notes

## 保持不变的研究主线

- Gymnasium 旋转倒立摆环境；
- Stable-Baselines3 PPO；
- 200 Hz 物理积分与 200 Hz 策略输出；
- `trig_hist4_act4` 28 维历史观测；
- 动力学、执行器、传感器与初值随机化；
- curriculum domain randomization；
- nominal/randomized 双评估、best checkpoint、VecNormalize；
- 7 维与 28 维 64×64 DAgger 风格策略蒸馏。

## 本次新增

`training_panel.py` 只是配置与运行界面，不改变 PPO 算法。面板通过 AST 精确替换 `config.py` 中的目标值；训练仍由原 `run.py` 和 SB3 PPO 完成。

`TrainingProgressCallback` 每隔 `PANEL["progress_update_freq"]` 输出一次：

```text
[PROGRESS] algorithm=PPO timesteps=... total=... percent=... fps=... eta_s=...
```

并保存 `runs/<experiment>/training_progress.json`，供面板或后续外部监控工具读取。

## 当前 PPO 基线

- `n_envs=8`；
- `n_steps=1024`；
- `batch_size=1024`；
- `gamma=0.9975`；
- Actor/Critic 均为 `(256, 256)`；
- observation 为 `trig_hist4_act4`；
- 训练与评估保持 15 s episode。
