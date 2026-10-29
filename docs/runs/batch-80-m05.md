# Batch 80 on m05

200 PPO runs on **3DBall** from batch seed 80: 50 configs, each with seeds 2 and 3, on both balls (`3DBall_b80` with the fixed ball, `3DBallRandomize_b80` with the random ball).

**In progress: 14 of 200 runs logged**, configs 0 to 6 with seed 2 on both balls. This page is updated when the batch ends.

m05 (`docs/hardware/machines.csv`), 2026-10-29, ml-agents 1.0.0 (our fork at `c080ea2`), torch 2.8.0+cpu, Python 3.10.11, Unity 2022.3.4f1, Windows 11 Pro 26200.

## How the runs were made

The build is `builds/3DBall.exe`: the 3DBall scene only (`Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity`), Windows 64-bit, default build options. The project's Build Settings list no scenes, so it was built from the command line (`Unity.exe -batchmode -executeMethod`) with a throwaway editor script that passes the scene to `BuildPipeline.BuildPlayer`.

**Burst AOT is off in this build.** Smart App Control is on on m05 and blocks Burst's compiler (`Burst.Compiler.IL`: "An Application Control policy has blocked this file"), so the first build failed. Setting `EnableBurstCompilation` to `false` in `ProjectSettings/BurstAotSettings_StandaloneWindows.json` (a file Unity writes on the first import, not tracked by the fork) fixed it. Burst only compiles code of Sentis, the inference backend. During training the actions come from Python and the physics from PhysX, so the runs should not change. Config 15 of this batch is config 20 of batch 46 (m02), which will check that.

The two balls ran at the same time, each in its own window, from the repository root with the ML-Agents environment active:

```powershell
python tools\paired_batch.py --name 3DBall_b80 --config config/ppo/3DBall.yaml --env-path builds\3DBall.exe --batch-seed 80 --seeds 2 3 --pair-with 3DBallRandomize_b80 --base-port 5005
python tools\paired_batch.py --name 3DBallRandomize_b80 --config config/ppo/3DBall_randomize.yaml --env-path builds\3DBall.exe --batch-seed 80 --seeds 2 3 --pair-with 3DBall_b80 --base-port 5105
```

Graphics were on, as in the README. m02's batches 46, 47 and 49 to 51 ran with `no_graphics: true`.

Duration is `total` from `results/<run_id>/run_logs/timers.json`, with two runs going at once. "Reached 80" is the run log's `reached_threshold`, a moving average of three summaries at 80 or more. "Steps to 95" is the first logged step with a mean reward of 95 or more, as in `analysis/datasets/outcomes.py`. So a run can reach 95 before it reaches 80.

## Smoke runs

Two runs of the base `config/ppo/3DBall.yaml` before the batch, at the same time and not logged: one in the editor (README, "Running a training run"; the editor script entered Play mode) and one with the build.

| run_id | environment | steps | duration (s) | reward_final | first step at 95 |
|---|---|---|---|---|---|
| m05_editor_smoke | Unity editor | 500,344 | 697.60 | 100.00 | 120,000 |
| m05_build_smoke | `builds/3DBall.exe` | 500,699 | 630.48 | 100.00 | 120,000 |

## Results so far

| ball | runs | reached 80 | reached 95 | steps to 95, median (IQR) | steps to 95, range | duration (min), median (range) |
|---|---|---|---|---|---|---|
| fixed | 7 | 6 | 6 | 114,000 (90,000 to 129,000) | 72,000 to 192,000 | 10.7 (9.0 to 13.7) |
| random | 7 | 6 | 6 | 132,000 (102,000 to 189,000) | 84,000 to 312,000 | 9.9 (9.1 to 13.6) |

| config | batch_size | buffer_size | learning_rate | hidden_units | num_layers | epsilon | beta | num_epoch | fixed: steps to 95 | random: steps to 95 |
|---|---|---|---|---|---|---|---|---|---|---|
| c00 | 64 | 2048 | 0.0001 | 128 | 3 | 0.2 | 0.005 | 8 | 72,000 | 84,000 |
| c01 | 32 | 4096 | 0.003 | 256 | 1 | 0.1 | 0.001 | 5 | 192,000 | 204,000 |
| c02 | 128 | 16384 | 0.003 | 128 | 2 | 0.3 | 0.005 | 5 | 132,000 | 120,000 |
| c03 | 64 | 4096 | 0.003 | 128 | 3 | 0.1 | 0.001 | 3 | 120,000 | 312,000 |
| c04 | 256 | 8192 | 0.001 | 64 | 3 | 0.1 | 0.001 | 5 | 108,000 | 144,000 |
| c05 | 128 | 4096 | 0.003 | 256 | 2 | 0.3 | 0.005 | 3 | not reached (final 0.31) | not reached (final 26.46) |
| c06 | 64 | 4096 | 0.0001 | 256 | 2 | 0.3 | 0.001 | 3 | 84,000 | 96,000 |

Config 5 failed on both balls. Of the six configs that reached 95 on both balls, the random ball took more steps in five (median 12,000 more).

Over batches 42 to 60, 89% to 100% of each batch's runs reached 95, with a median of 108,000 to 144,000 steps on the fixed ball and 120,000 to 156,000 on the random ball. Seven configs are too few to compare batch 80 with them yet.


