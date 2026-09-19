# First training runs

Two runs on **3DBall**: same environment, different algorithms (PPO, SAC).

**Not a controlled comparison.** The two configs differ in five fields (`trainer_type`, `max_steps`, `hidden_units`, `buffer_size`, `learning_rate_schedule`). Nothing here says which algorithm is better, only testing the setup.

Both reached `max_steps` and solved the task (3DBall rewards cap at 100).

Duration is `total` from `run_logs/timers.json`. Reward is the last `Environment/Cumulative Reward` in the TensorBoard event file.

m01, 2026-09-19, ml-agents 1.0.0, torch 2.14.0+cpu, Unity 2022.3.4f1.

| run_id | algorithm | config | steps | duration (s) | reward_final |
|---|---|---|---|---|---|
| smoke01 | ppo | `config/ppo/3DBall.yaml` | 499,348 | 436.15 | 100.00 |
| sac3dball01 | sac | `config/sac/3DBall.yaml` | 191,353 | 222.02 | 100.00 |
