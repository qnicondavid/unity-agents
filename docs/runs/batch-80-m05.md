# Batch 80 on m05

200 PPO runs on **3DBall** from batch seed 80: 50 configs, each with seeds 2 and 3, on both balls (`3DBall_b80` with the fixed ball, `3DBallRandomize_b80` with the random ball).

All 200 runs finished and were logged on the first try.

m05 (`docs/hardware/machines.csv`), 2026-10-29 to 2026-10-30, ml-agents 1.0.0 (our fork at `c080ea2`), torch 2.8.0+cpu, Python 3.10.11, Unity 2022.3.4f1, Windows 11 Pro 26200.

## How the runs were made

The build is `builds/3DBall.exe`: the 3DBall scene only (`Assets/ML-Agents/Examples/3DBall/Scenes/3DBall.unity`), Windows 64-bit, default build options. The project's Build Settings list no scenes, so it was built from the command line (`Unity.exe -batchmode -executeMethod`) with a throwaway editor script that passes the scene to `BuildPipeline.BuildPlayer`.

**Burst AOT is off in this build.** Smart App Control is on on m05 and blocks Burst's compiler (`Burst.Compiler.IL`: "An Application Control policy has blocked this file"), so the first build failed. Setting `EnableBurstCompilation` to `false` in `ProjectSettings/BurstAotSettings_StandaloneWindows.json` (a file Unity writes on the first import, not tracked by the fork) fixed it. Burst only compiles code of Sentis, the inference backend. During training the actions come from Python and the physics from PhysX, so the runs should not change. Config 15 of this batch is config 20 of batch 46, which ran on m02 with `no_graphics: true`. Its four runs logged the same values at every step as those of batch 46, so neither Burst nor the graphics changed them.

The two balls ran at the same time, each in its own window, from the repository root with the ML-Agents environment active:

```powershell
python tools\paired_batch.py --name 3DBall_b80 --config config/ppo/3DBall.yaml --env-path builds\3DBall.exe --batch-seed 80 --seeds 2 3 --pair-with 3DBallRandomize_b80 --base-port 5005
python tools\paired_batch.py --name 3DBallRandomize_b80 --config config/ppo/3DBall_randomize.yaml --env-path builds\3DBall.exe --batch-seed 80 --seeds 2 3 --pair-with 3DBall_b80 --base-port 5105
```

Graphics were on, as in the README. m02's batches 46, 47 and 49 to 51 ran with `no_graphics: true`.

Duration is `total` from `results/<run_id>/run_logs/timers.json`, with two runs going at once. Standby paused both runs of configs 7 and 14 with seed 2 (82 and 134 minutes between two summaries, against 14 seconds normally), so the durations leave those four runs out. "Reached 80" is the run log's `reached_threshold`, a moving average of three summaries at 80 or more. "Steps to 95" is the first logged step with a mean reward of 95 or more, as in `analysis/datasets/outcomes.py`. So a run can reach 95 before it reaches 80.

## Smoke runs

Two runs of the base `config/ppo/3DBall.yaml` before the batch, at the same time and not logged: one in the editor (README, "Running a training run"; the editor script entered Play mode) and one with the build.

| run_id | environment | steps | duration (s) | reward_final | first step at 95 |
|---|---|---|---|---|---|
| m05_editor_smoke | Unity editor | 500,344 | 697.60 | 100.00 | 120,000 |
| m05_build_smoke | `builds/3DBall.exe` | 500,699 | 630.48 | 100.00 | 120,000 |

## Results

| ball | runs | reached 80 | reached 95 | steps to 95, median (IQR) | steps to 95, range | duration (min), median (range) |
|---|---|---|---|---|---|---|
| fixed | 100 | 99 | 98 | 132,000 (96,000 to 189,000) | 36,000 to 312,000 | 9.6 (7.0 to 35.4) |
| random | 100 | 98 | 98 | 156,000 (132,000 to 237,000) | 48,000 to 420,000 | 9.5 (7.1 to 35.8) |

Four runs did not reach 95, all with seed 2: config 5 on both balls, config 37 on the fixed ball (it reached 80 and ended at 90.05) and config 8 on the random ball. With seed 3 all three configs reached 95, so the two seeds agree on `reached_95` for 48 of the 50 configs on each ball. Where both seeds reached 95, they were a median of 24,000 steps apart (up to 168,000), and the same in 17 of 96 cases.

25 runs that reached 95 ended below it (12 fixed, 13 random), most of them only just (median 92.4). `3DBall_b80_c26_s2` reached 95 at 72,000 steps and ended at 11.31.

In the 97 config and seed pairs where both balls reached 95, the random ball took more steps in 59, the same in 20 and fewer in 18 (a median of 12,000 more). Config 25 was the fastest (48,000 to 72,000 steps on its four runs), and configs 10, 18 and 44 the slowest (a mean of 276,000 to 282,000).

Over batches 42 to 60, 89% to 100% of each batch's runs reached 95, with a median of 108,000 to 144,000 steps on the fixed ball and 120,000 to 156,000 on the random ball. Batch 80 is within these ranges: 98% on both balls, with medians of 132,000 and 156,000.

Steps to 95 of every run:

| config | batch_size | buffer_size | learning_rate | hidden_units | num_layers | epsilon | beta | num_epoch | fixed, seed 2 | fixed, seed 3 | random, seed 2 | random, seed 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c00 | 64 | 2048 | 0.0001 | 128 | 3 | 0.2 | 0.005 | 8 | 72,000 | 84,000 | 84,000 | 48,000 |
| c01 | 32 | 4096 | 0.003 | 256 | 1 | 0.1 | 0.001 | 5 | 192,000 | 204,000 | 204,000 | 192,000 |
| c02 | 128 | 16384 | 0.003 | 128 | 2 | 0.3 | 0.005 | 5 | 132,000 | 132,000 | 120,000 | 120,000 |
| c03 | 64 | 4096 | 0.003 | 128 | 3 | 0.1 | 0.001 | 3 | 120,000 | 288,000 | 312,000 | 168,000 |
| c04 | 256 | 8192 | 0.001 | 64 | 3 | 0.1 | 0.001 | 5 | 108,000 | 144,000 | 144,000 | 192,000 |
| c05 | 128 | 4096 | 0.003 | 256 | 2 | 0.3 | 0.005 | 3 | not reached (final 0.31) | 168,000 | not reached (final 26.46) | 420,000 |
| c06 | 64 | 4096 | 0.0001 | 256 | 2 | 0.3 | 0.001 | 3 | 84,000 | 72,000 | 96,000 | 72,000 |
| c07 | 256 | 4096 | 0.0003 | 64 | 2 | 0.1 | 0.01 | 8 | 96,000 | 96,000 | 132,000 | 144,000 |
| c08 | 32 | 4096 | 0.001 | 256 | 3 | 0.3 | 0.001 | 3 | 240,000 | 72,000 | not reached (final 44.55) | 264,000 |
| c09 | 32 | 2048 | 0.001 | 128 | 1 | 0.2 | 0.005 | 3 | 48,000 | 72,000 | 108,000 | 108,000 |
| c10 | 256 | 16384 | 0.003 | 256 | 3 | 0.1 | 0.01 | 3 | 288,000 | 276,000 | 300,000 | 240,000 |
| c11 | 128 | 8192 | 0.0003 | 256 | 2 | 0.2 | 0.001 | 3 | 84,000 | 108,000 | 132,000 | 132,000 |
| c12 | 256 | 2048 | 0.003 | 64 | 3 | 0.3 | 0.001 | 5 | 36,000 | 48,000 | 72,000 | 144,000 |
| c13 | 64 | 8192 | 0.001 | 256 | 2 | 0.1 | 0.005 | 3 | 192,000 | 192,000 | 156,000 | 132,000 |
| c14 | 32 | 4096 | 0.0001 | 256 | 2 | 0.2 | 0.005 | 8 | 132,000 | 144,000 | 144,000 | 144,000 |
| c15 | 64 | 2048 | 0.001 | 64 | 3 | 0.2 | 0.005 | 5 | 204,000 | 132,000 | 168,000 | 156,000 |
| c16 | 128 | 8192 | 0.0003 | 64 | 1 | 0.3 | 0.001 | 3 | 108,000 | 108,000 | 108,000 | 96,000 |
| c17 | 32 | 16384 | 0.0003 | 256 | 1 | 0.3 | 0.01 | 5 | 132,000 | 144,000 | 132,000 | 144,000 |
| c18 | 64 | 16384 | 0.003 | 128 | 3 | 0.1 | 0.01 | 5 | 276,000 | 252,000 | 348,000 | 252,000 |
| c19 | 32 | 2048 | 0.0003 | 256 | 2 | 0.2 | 0.005 | 8 | 72,000 | 132,000 | 204,000 | 312,000 |
| c20 | 64 | 2048 | 0.0001 | 64 | 3 | 0.3 | 0.01 | 3 | 72,000 | 60,000 | 84,000 | 84,000 |
| c21 | 32 | 8192 | 0.003 | 64 | 3 | 0.1 | 0.01 | 5 | 252,000 | 168,000 | 288,000 | 264,000 |
| c22 | 64 | 8192 | 0.0003 | 256 | 3 | 0.2 | 0.005 | 8 | 132,000 | 84,000 | 180,000 | 252,000 |
| c23 | 128 | 2048 | 0.003 | 64 | 2 | 0.2 | 0.005 | 8 | 120,000 | 72,000 | 312,000 | 156,000 |
| c24 | 128 | 8192 | 0.0003 | 256 | 1 | 0.1 | 0.01 | 3 | 180,000 | 204,000 | 204,000 | 204,000 |
| c25 | 256 | 2048 | 0.0003 | 64 | 3 | 0.2 | 0.005 | 5 | 48,000 | 60,000 | 72,000 | 60,000 |
| c26 | 128 | 8192 | 0.001 | 128 | 3 | 0.3 | 0.01 | 3 | 72,000 | 96,000 | 192,000 | 132,000 |
| c27 | 128 | 4096 | 0.0003 | 128 | 2 | 0.1 | 0.005 | 3 | 108,000 | 108,000 | 144,000 | 132,000 |
| c28 | 64 | 8192 | 0.0003 | 256 | 1 | 0.2 | 0.001 | 5 | 96,000 | 108,000 | 108,000 | 156,000 |
| c29 | 32 | 8192 | 0.003 | 256 | 1 | 0.1 | 0.01 | 5 | 264,000 | 180,000 | 264,000 | 312,000 |
| c30 | 256 | 16384 | 0.003 | 128 | 3 | 0.1 | 0.01 | 5 | 192,000 | 240,000 | 276,000 | 360,000 |
| c31 | 64 | 8192 | 0.003 | 128 | 3 | 0.2 | 0.01 | 3 | 312,000 | 276,000 | 228,000 | 144,000 |
| c32 | 128 | 8192 | 0.001 | 64 | 2 | 0.2 | 0.001 | 3 | 84,000 | 96,000 | 84,000 | 96,000 |
| c33 | 32 | 16384 | 0.0003 | 256 | 1 | 0.2 | 0.001 | 3 | 192,000 | 204,000 | 204,000 | 204,000 |
| c34 | 64 | 16384 | 0.001 | 256 | 2 | 0.2 | 0.01 | 8 | 228,000 | 180,000 | 144,000 | 216,000 |
| c35 | 32 | 4096 | 0.0001 | 256 | 2 | 0.1 | 0.001 | 5 | 120,000 | 108,000 | 132,000 | 228,000 |
| c36 | 128 | 16384 | 0.0003 | 128 | 1 | 0.2 | 0.005 | 5 | 180,000 | 180,000 | 180,000 | 204,000 |
| c37 | 32 | 2048 | 0.003 | 128 | 3 | 0.2 | 0.005 | 3 | not reached (final 90.05) | 216,000 | 360,000 | 348,000 |
| c38 | 128 | 2048 | 0.0001 | 128 | 2 | 0.2 | 0.005 | 3 | 108,000 | 84,000 | 84,000 | 84,000 |
| c39 | 256 | 4096 | 0.001 | 256 | 2 | 0.1 | 0.001 | 5 | 144,000 | 108,000 | 252,000 | 168,000 |
| c40 | 64 | 4096 | 0.001 | 256 | 2 | 0.2 | 0.005 | 3 | 132,000 | 108,000 | 120,000 | 144,000 |
| c41 | 128 | 16384 | 0.0001 | 256 | 2 | 0.1 | 0.005 | 3 | 288,000 | 240,000 | 252,000 | 276,000 |
| c42 | 32 | 16384 | 0.0001 | 128 | 3 | 0.2 | 0.001 | 3 | 132,000 | 156,000 | 192,000 | 156,000 |
| c43 | 64 | 2048 | 0.003 | 128 | 1 | 0.3 | 0.01 | 5 | 72,000 | 48,000 | 264,000 | 252,000 |
| c44 | 32 | 2048 | 0.003 | 128 | 1 | 0.1 | 0.001 | 5 | 312,000 | 156,000 | 312,000 | 324,000 |
| c45 | 256 | 8192 | 0.001 | 256 | 1 | 0.1 | 0.001 | 3 | 180,000 | 180,000 | 216,000 | 192,000 |
| c46 | 128 | 2048 | 0.0001 | 128 | 1 | 0.1 | 0.01 | 3 | 156,000 | 144,000 | 156,000 | 144,000 |
| c47 | 128 | 2048 | 0.0003 | 64 | 1 | 0.1 | 0.01 | 5 | 120,000 | 108,000 | 108,000 | 108,000 |
| c48 | 32 | 16384 | 0.0001 | 64 | 1 | 0.3 | 0.001 | 8 | 144,000 | 132,000 | 144,000 | 144,000 |
| c49 | 256 | 4096 | 0.0003 | 256 | 3 | 0.2 | 0.005 | 8 | 120,000 | 60,000 | 72,000 | 156,000 |
