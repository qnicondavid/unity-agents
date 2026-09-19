# Group README notes

## Contents

- [Group 1](#group-1)
- [Group 16](#group-16)
- [Group 8](#group-8)
- [Group 11](#group-11)
- [Group 5](#group-5)
- [Group 6](#group-6)
- [Group 7](#group-7)
- [Group 9](#group-9)
- [Group 14](#group-14)
- [Group 15](#group-15)

Six groups submitted only the unedited template (12-13 bytes, just the heading) and no data:
**Groups 2, 3, 4, 10, 12, 13.** Nothing to read there.

---

## Group 1

Files: 3 CSVs (47 / 24 / 23 rows), all per-run. README 439 words **plus a
separate `Datafile.md` data dictionary, 477 words** (the only group that
wrote one).

**Collected**

PPO only, two environments (Pyramids 24 runs, Worm 23).
Claim is 5 configs x 2 seeds per env, the data shows 4 distinct hyperparameter combinations x 6 runs each.

**Columns defined**

`Datafile.md` gives every metric a meaning, a unit, how it was measured and what it is for (five columns per metric).
This could be used in our project also.

**Measurement method**

Stated per metric: `wallclock_seconds_total` from start/end times,
`cpu_cores_logical` from `os.cpu_count()`, `ram_total_gb` from psutil,
`steps_to_threshold` by detecting when reward first crosses a threshold,
`plateau_reward` as the mean of the last 10 logs over 6 baseline runs.

**Caveats they admit**

No known-issues section.
`Datafile.md` says `plateau_reward` is "added only for the 4 non baseline configurations".
It is present on all 47 rows, one value per environment, though Worm's appears at two precisions (`939.0479570661272` on 16 rows, `939.0479` on 7).

---

## Group 16

Files: 1 CSV (200 rows, 50 columns). README 1,400 words across 347 lines - the most thorough in the repo.

**Collected**

Claims 67 runs across 6 environments, PPO and SAC, January 2026.
Data has 200 rows across 11 environments (adds BigWallJump, FoodCollector,
Hallway, Pyramids, Worm). 171 PPO, 29 SAC.

**Columns defined**

Every one of the 50 columns has a section giving type, range and context,
grouped into 15 themed categories. Separates PPO-only from SAC-only
hyperparameters explicitly. A very good example of documentation.

**Measurement method**

Not stated. Contrast with Group 1, which gives
a "how measured" entry per metric.

**Caveats they admit**

No known-issues section.
3DBall `final_mean_reward` exceeds 100 in 9 rows. ML-Agents caps 3DBall reward at 100.
Correlation between `steps` and `training_duration_seconds` is 0.24.
README quality and data quality are independent.
Documentation should be judged on whether its claims check out against the file, not on how thorough it looks.

---

## Group 8

Files: 2 CSVs (432 rows each), **semicolon-delimited**. README 817 words.
Only group in the repo using a `machine_id`.

**Collected**

432 PPO/SAC runs on 3DBall, all in `training_data_1/`. 364 PPO, 68 SAC.
Randomised `buffer_size`, `hidden_units`, `num_layers` for both algorithms,
plus the SAC-specific parameters for SAC runs.

**Columns defined**

44-row table giving name, type, unit and which group each column belongs to
(common/hardware/yaml/sac/ppo/final).

**Measurement method**

Convergence is defined explicitly: `mean_reward >= 100` AND
`std_reward / mean_reward <= 1`, first window satisfying both. If no window
qualifies, `steps_to_convergence` and `time_to_convergence` stay NA.
`max_steps` is 2,000,000 or 10,000,000

**Caveats they admit**

SAC data collection "took too long, in possible future datasets it won't
be included". And their workflow: data lands in a fork first, is merged by
pull request, never deleted or overwritten, new runs go to consecutive
`training_data_N` directories.

---

## Group 11

Files: 3 CSVs (31,867 / 15,892 / 1,000 rows). README 443 words.

**Collected**

PushBlock only, PPO only. 31,867 training runs in the largest file, plus a
1,000-row earlier sample with identical columns, plus a 15,892-row reward
time series covering 2,012 runs.

**Columns defined**

28 columns, one-line description each. No units given, and no statement of how
anything was measured.

**Measurement method**

Not stated.

**Caveats they admit**

No known-issues section. All 31,867 rows report `total_ram_mb` = 11700.625 and `cpu_cores` = 6 (one machine).
And eight of fifteen hyperparameters are constant
(`learning_rate`, `beta`, `epsilon`, `lambd`, `gamma`, `max_steps`,
`strength`, `keep_checkpoints`). What varies is `buffer_size`, `batch_size`,
`hidden_units`, `time_horizon`, `num_epoch`, `num_layers`, `summary_freq`.

---

## Group 5

Files: 1 CSV (288 rows). README 443 words.

**Collected**

288 runs, PPO only: `trainer_type` is `ppo` on every row despite the column
description mentioning SAC.
There is no `env_name`, no `environment`,
nothing identifying which Unity environment any run used.
No reward, no convergence, no loss, no
entropy. The only outcome-like columns are `total_duration` and
`total_steps`.

**Columns defined**

47 documented, 57 present. Ten columns in the file are not in the README:
`host_os_name`, `is_docker_used`, `disk_total_gb`, `disk_free_gb`,
`disk_filesystem`, `checkpoint_interval`, `keep_checkpoints`,
`even_checkpoints`, `beta_schedule`, `epsilon_schedule`.

Hardware capture is the most thorough in the repo: vendor, architecture,
physical and logical cores, base frequency, total and available RAM, RAM
type, GPU count and two GPU slots, plus the undocumented disk and docker
columns.

**Measurement method**

Not stated.

**Caveats they admit**

None.

---

## Group 6

Files: 2 CSVs (3,913 and 28 rows, the second named `_old`). README 506 words.

**Collected**

3,913 rows across 6 environments, overwhelmingly 3DBall PPO (3,736 of 3,913).
Only 10 SAC rows. Basic has 46, Crawler 31, and BigWallJump, Hallway and
PushBlock 30 each.

**Columns defined**

33 documented, 34 present (`notes`, which is null in all 3,913 rows).
`run_reached_threshold` is **NaN on every row**, despite being documented as a bool.
No hardware columns at all.

**Measurement method**

Partly stated. `threshold_value` is per-environment (6 distinct, one each),
so thresholds are environment-specific rather than a single global number.
Reward columns distinguish early / mean / best / final with a matching
`_step` column for each, so when each was measured is explicit.

**Caveats they admit**

`combined_results_old.csv` is excluded from their
analysis because of "incompletion and inaccuracies found in data from early
versions of our testing runner".

With no hardware columns, the data cannot contribute anything to a
model that uses machine characteristics as features, and the CPU/RAM
percentages are uninterpretable without the machines they were measured on.

---

## Group 7

Files: 3 CSVs (8,389 / 8,389 / 1,237 rows), named `old_data`, `Data`,
`new_data`. README 444 words.

**Collected**

Three CSVs, all PPO, no SAC.
`new_data.csv` (1,237 × 24) is the one they say to use.
24 columns in all three: hardware, PPO hyperparameters, and one outcome block (`cumulative_reward_mean`, `time_elapased_seconds`, `max_steps`).
No environment column, so every run is assumed to be the same env without saying which.

**Columns defined**

24 documented, 24 present, zero discrepancies in either direction
One of five groups where the documentation matches the data exactly (1, 7, 8, 11, 16).
But the README refers to `old_dataset.csv`, which does not exist; the files are `Data.csv`, `new_data.csv`, `old_data.csv`.

**Measurement method**

"How values are recorded", tags each column with its source of origin: *User*, *Program*, *System Query*, *Training Configuration File*, *Data Log*.

**Caveats they admit**

"old_dataset.csv originally contained many more training runs but had significantly skewed data.
They do not say what the skew was, how they detected it, or what rule removed the 7,152 rows.

---

## Group 9

Files: 1 CSV (164 rows). README 215 words.

**Collected**

One file, `dataset.csv`, 164 rows × 19 columns, 39.9 KB.
Every row is SoccerTwos and every row is `poca`.
One of only two groups running MA-POCA self-play instead of PPO or SAC (Group 14 is the other), which means their rows cannot be pooled with the PPO/SAC groups regardless of schema (the reward semantics are different).
**There are no hardware columns at all.**

**Columns defined**

17 documented, 19 present.
`mean_group_entropy` is documented but does not exist. `algorithm`, `episode_length` and `mean_value_loss` exist but are not documented.

**Measurement method**

No statement of how anything was captured, by what script, at what point in the run, or on what machine.

**Caveats they admit**

No limitations section, no known-issues note, nothing about the single environment or the missing hardware.

---

## Group 14

Files: **20 CSVs** (more than any other group). README **191 words**
The worst documentation-to-data ratio in the repo.

**Collected**

The 11 uppercase `training_data_*` files share a 23-column wide format (2,216 rows).
Four lowercase files use a 14-column format with `run_id`/`environment`/`algorithm` but no hyperparameters and no losses (2,081 rows), and one of those four uses `timestamp` where the others use `time_elapsed`.
The `SCTWRUN2-additional-data/` subfolder holds five raw TensorBoard exports, `Wall time,Step,Value`, one file per metric (1,968 rows).
No hardware columns anywhere in any schema.

**Columns defined**

15 metrics plus 8 hyperparameters, which match the 23-column files exactly.
Where `environment` does exist it says `multi-agent` or literally `unknown`, so it never names the environment either.

**Measurement method**

Not stated.

**Caveats they admit**

One: `gpu_usage_avg_mb` "can not be calculated if your system has an integrated graphics card".

---

## Group 15

Files: **no readable CSVs** (only `.zip` and `.7z` archives, 30 MB of the
repo's 45). README 874 words.

**Collected**

Two archives.
`Group15_data.zip` (8.9 MB) holds `static.csv` / `summary.csv` / `training.csv` (**1,436 runs**).
`Group15_data.7z` (21.7 MB) holds `static_rows.csv` / `summary_rows.csv` / `training_rows.csv` (**5,710 runs, 362,540 training rows**).
The README describes only the zip ("From ZIP file you will get 3 csv files") and never mentions the 7z, yet the 5,710 figure it advertises is the 7z's.
Every one of the zip's 1,436 runs is also in the 7z, so the zip is a strict subset and can be ignored.
PPO 5,584 / SAC 126, across 11 environments, of which 3DBall and 3DBallHard are 79%

**Columns defined**

`static` is per-run config plus hardware capacity (28 cols), `training` is the per-step time series (23 cols), `summary` is per-run aggregates (19 documented).
Two README rows, `beta` and `episode time`, have the description cell left blank.

**Measurement method**

Not stated anywhere.

**Caveats they admit**

Two. The compression note, and `gpu_usage_percent` "if -1, gpu wasnt used".

It is the only group with a genuine relational design separating capacity, per-step consumption and per-run outcome, the only one with both algorithms across 11 environments at this scale, and the only one whose three tables join without loss.
The limit is hardware. `physical_cores` is 8 on all 5,710 rows and `cuda_version` is blank on all 5,710.
The six apparent hardware combinations collapse to **four actual machines**.

---

