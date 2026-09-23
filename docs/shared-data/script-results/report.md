# Shared data survey

- groups with at least one readable CSV: **9**
- CSV files: **36**
- total rows: **78,590**
- distinct column names: **209**
- names used by more than one group: **35**
- names used by exactly one group: **174**

## Granularity split

- **per-step**: 23 files, 22,385 rows
- **per-run**: 12 files, 55,773 rows
- **unknown**: 1 files, 432 rows

## Same quantity, different spelling

**ram_total**
  - `ram_gb`: Group 16
  - `ram_mb`: Group 8
  - `ram_total_bytes`: Group 5
  - `ram_total_gb`: Group 1
  - `total_ram`: Group 7
  - `total_ram_mb`: Group 11

**cpu_cores_logical**
  - `cpu_cores_logical`: Group 1
  - `cpu_logical_cores`: Group 8
  - `cpu_threads`: Group 5

**cpu_clock**
  - `cpu_clock_ghz`: Group 8
  - `cpu_frequency`: Group 7
  - `cpu_frequency_khz`: Group 5

**gpu_model**
  - `gpu_0_name`: Group 5
  - `gpu_1_name`: Group 5
  - `gpu_model`: Group 7
  - `gpu_name`: Group 16

**gpu_vram**
  - `gpu_0_vram_gb`: Group 5
  - `gpu_1_vram_gb`: Group 5
  - `gpu_memory_gb`: Group 16

**os_name**
  - `host_os_name`: Group 5
  - `operating_system`: Group 7
  - `os`: Group 16
  - `os_name`: Group 8

**ram_used_avg**
  - `average_ram`: Group 6
  - `avg_ram_usage`: Group 8
  - `memory_usage_avg_mb`: Group 14

**ram_used_peak**
  - `memory_usage_peak_mb`: Group 14
  - `peak_ram_mb`: Group 16
  - `peak_ram_usage`: Group 8

**cpu_used_avg**
  - `average_cpu`: Group 6
  - `avg_cpu_usage`: Group 8
  - `cpu_usage_avg_percent`: Group 14

**cpu_used_peak**
  - `cpu_usage_peak_percent`: Group 14
  - `peak_cpu_usage`: Group 8

**run_duration**
  - `duration_sec`: Group 11
  - `time_elapased_seconds`: Group 7
  - `time_elapsed`: Group 14
  - `total_duration`: Group 5
  - `total_time`: Group 6
  - `train_duration_s`: Group 8
  - `training_duration_seconds`: Group 16
  - `training_time`: Group 9
  - `wallclock_seconds_total`: Group 1

**time_to_threshold**
  - `time_to_convergence`: Group 8
  - `time_to_threshold`: Group 6

**steps_to_threshold**
  - `steps_to_convergence`: Group 8
  - `steps_to_threshold`: Group 1, Group 6

**algorithm**
  - `algo`: Group 8
  - `algo_name`: Group 1
  - `algorithm`: Group 14, Group 16, Group 6, Group 9
  - `drl_algorithm`: Group 7
  - `trainer_type`: Group 14, Group 5

**environment**
  - `env_name`: Group 1, Group 8, Group 9
  - `enviroment`: Group 16
  - `environment`: Group 14, Group 6

**batch_size**
  - `batch_size`: Group 1, Group 11, Group 14, Group 16, Group 5, Group 6, Group 7, Group 8
  - `bath_size`: Group 9

**lambda**
  - `lambd`: Group 11, Group 14, Group 16, Group 5, Group 8
  - `lambda`: Group 7, Group 9

**num_epoch**
  - `epoch`: Group 9
  - `epochs`: Group 6
  - `num_epoch`: Group 11, Group 14, Group 16, Group 5, Group 7, Group 8

**num_layers**
  - `nn_arch_depth`: Group 1
  - `num_layers`: Group 11, Group 16, Group 5, Group 7, Group 8

**gamma**
  - `extrinsic_gamma`: Group 5
  - `gamma`: Group 11, Group 16, Group 7, Group 8, Group 9

**strength**
  - `extrinsic_strength`: Group 5
  - `strength`: Group 11, Group 16, Group 8

**num_agents**
  - `num_agents`: Group 6
  - `num_parallel_agents`: Group 16

**run_id**
  - `runNr`: Group 11
  - `run_id`: Group 1, Group 11, Group 14, Group 5, Group 6, Group 7, Group 8, Group 9

**step**
  - `Step`: Group 14
  - `nrSteps`: Group 11
  - `step`: Group 14

**policy_loss**
  - `final_mean_policy_loss`: Group 16
  - `losses_policy_loss`: Group 14
  - `mean_policy_loss`: Group 9
  - `p_loss_mean`: Group 6

**value_loss**
  - `final_mean_value_loss`: Group 16
  - `losses_value_loss`: Group 14
  - `mean_value_loss`: Group 9
  - `v_loss_mean`: Group 6

**entropy**
  - `entropy_mean`: Group 6
  - `final_mean_entropy`: Group 16
  - `mean_entropy`: Group 9
  - `policy_entropy`: Group 14

**reward_mean**
  - `cumulative_reward_mean`: Group 7
  - `episodic_reward_mean`: Group 1
  - `meanReward`: Group 11
  - `mean_reward`: Group 14
  - `reward_mean`: Group 6

**reward_final**
  - `final_cumulative_mean_reward`: Group 16
  - `final_mean_reward`: Group 16, Group 8
  - `final_perf`: Group 1
  - `final_reward_mean`: Group 6

**reward_std**
  - `final_std_reward`: Group 8
  - `std_reward`: Group 14


## Non-comma delimiters

- `Group 8/training_data_1/encoded_dataset.csv`: semicolon
- `Group 8/training_data_1/main.csv`: semicolon

## Needs attention

- Group 15\Group15_data.7z: archive, needs unpacking before it counts as data
- Group 15\Group15_data.zip: archive, needs unpacking before it counts as data

## Columns used by exactly one group

Candidates for further synonyms, or genuinely unique measurements.

- `ELO`: Group 9
- `Step`: Group 14
- `Value`: Group 14
- `Wall time`: Group 14
- `algo`: Group 8
- `algo_name`: Group 1
- `algo_sac`: Group 8
- `average_cpu`: Group 6
- `average_ram`: Group 6
- `avg_cpu_usage`: Group 8
- `avg_ram_usage`: Group 8
- `bath_size`: Group 9
- `behavioral_cloning`: Group 5
- `best_reward`: Group 6
- `best_reward_before_timeout`: Group 6
- `best_reward_step`: Group 6
- `beta_schedule`: Group 5
- `buffer_init_steps`: Group 8
- `checkpoint_interval`: Group 5
- `config_name`: Group 11
- `converged`: Group 8
- `cpu_architecture`: Group 5
- `cpu_avg_python`: Group 11
- `cpu_avg_unity`: Group 11
- `cpu_clock_ghz`: Group 8
- `cpu_cores_logical`: Group 1
- `cpu_frequency`: Group 7
- `cpu_frequency_khz`: Group 5
- `cpu_logical_cores`: Group 8
- `cpu_model`: Group 7
- `cpu_peak_python`: Group 11
- `cpu_peak_unity`: Group 11
- `cpu_physical_cores`: Group 8
- `cpu_threads`: Group 5
- `cpu_usage_avg_mb`: Group 14
- `cpu_usage_avg_percent`: Group 14
- `cpu_usage_peak_mb`: Group 14
- `cpu_usage_peak_percent`: Group 14
- `cpu_vendor`: Group 5
- `cumulative_reward_mean`: Group 7
- `deterministic`: Group 5
- `disk_filesystem`: Group 5
- `disk_free_gb`: Group 5
- `disk_total_gb`: Group 5
- `drl_algorithm`: Group 7
- `duration_sec`: Group 11
- `early_reward_mean`: Group 6
- `early_reward_mean_step`: Group 6
- `efficiency_score`: Group 9
- `entropy_mean`: Group 6
- `entropy_mean_step`: Group 6
- `enviroment`: Group 16
- `episode_length`: Group 9
- `episodic_reward_mean`: Group 1
- `epoch`: Group 9
- `epochs`: Group 6
- `epsilon_schedule`: Group 5
- `even_checkpoints`: Group 5
- `extrinsic_gamma`: Group 5
- `extrinsic_strength`: Group 5
- `final_cumulative_mean_reward`: Group 16
- `final_cumulative_sum_reward`: Group 16
- `final_learning_rate_mean`: Group 16
- `final_learning_rate_sum`: Group 16
- `final_mean_cont_entropy_coeff`: Group 16
- `final_mean_entropy`: Group 16
- `final_mean_extrinsic_value_estimate`: Group 16
- `final_mean_policy_loss`: Group 16
- `final_mean_q1_loss`: Group 16
- `final_mean_q2_loss`: Group 16
- `final_mean_value_loss`: Group 16
- `final_perf`: Group 1
- `final_reward_mean`: Group 6
- `final_reward_mean_step`: Group 6
- `final_std_reward`: Group 8
- `final_sum_cont_entropy_coeff`: Group 16
- `final_sum_entropy`: Group 16
- `final_sum_extrinsic_value_estimate`: Group 16
- `final_sum_policy_loss`: Group 16
- `final_sum_q1_loss`: Group 16
- `final_sum_q2_loss`: Group 16
- `final_sum_reward`: Group 16
- `final_sum_value_loss`: Group 16
- `goal_conditioning_type`: Group 5
- `gpu_0_name`: Group 5
- `gpu_0_vendor`: Group 5
- `gpu_0_vram_gb`: Group 5
- `gpu_1_name`: Group 5
- `gpu_1_vendor`: Group 5
- `gpu_1_vram_gb`: Group 5
- `gpu_available`: Group 16
- `gpu_count`: Group 5
- `gpu_memory_gb`: Group 16
- `gpu_model`: Group 7
- `gpu_name`: Group 16
- `gpu_usage_avg_mb`: Group 14
- `gpu_usage_peak_mb`: Group 14
- `group_cumulative_reward`: Group 9
- `group_id`: Group 5
- `host_os_name`: Group 5
- `init_path`: Group 5
- `is_docker_used`: Group 5
- `losses_policy_loss`: Group 14
- `losses_value_loss`: Group 14
- `machine_id`: Group 8
- `meanReward`: Group 11
- `mean_entropy`: Group 9
- `mean_policy_loss`: Group 9
- `mean_reward`: Group 14
- `mean_value_loss`: Group 9
- `memory`: Group 5
- `memory_usage_avg_mb`: Group 14
- `memory_usage_peak_mb`: Group 14
- `nn_arch_depth`: Group 1
- `notes`: Group 6
- `nrSteps`: Group 11
- `num_agents`: Group 6
- `num_parallel_agents`: Group 16
- `operating_system`: Group 7
- `os`: Group 16
- `os_name`: Group 8
- `os_name_Windows`: Group 8
- `os_name_macOS`: Group 8
- `p_loss_mean`: Group 6
- `p_loss_mean_step`: Group 6
- `peak_cpu_usage`: Group 8
- `peak_ram_mb`: Group 16
- `peak_ram_usage`: Group 8
- `plateau_reward`: Group 1
- `policy_entropy`: Group 14
- `ram_available_bytes`: Group 5
- `ram_avg_python`: Group 11
- `ram_avg_unity`: Group 11
- `ram_gb`: Group 16
- `ram_mb`: Group 8
- `ram_peak_python`: Group 11
- `ram_peak_unity`: Group 11
- `ram_total_bytes`: Group 5
- `ram_total_gb`: Group 1
- `ram_type`: Group 5
- `reward_mean`: Group 6
- `reward_mean_step`: Group 6
- `runNr`: Group 11
- `run_log_file`: Group 8
- `run_reached_threshold`: Group 6
- `save_replay_buffer`: Group 8
- `self_play`: Group 5
- `shared_critic`: Group 5
- `std_reward`: Group 14
- `step`: Group 14
- `step_interval`: Group 6
- `step_of_best_reward`: Group 6
- `steps_per_update`: Group 8
- `steps_to_convergence`: Group 8
- `threaded`: Group 5
- `threshold_value`: Group 6
- `threshold_version`: Group 6
- `timeUnit`: Group 11
- `time_elapased_seconds`: Group 7
- `time_elapsed`: Group 14
- `time_end`: Group 5
- `time_start`: Group 5
- `time_to_convergence`: Group 8
- `time_to_threshold`: Group 6
- `total_duration`: Group 5
- `total_ram`: Group 7
- `total_ram_mb`: Group 11
- `total_time`: Group 6
- `train_duration_s`: Group 8
- `training_duration_seconds`: Group 16
- `training_time`: Group 9
- `v_loss_mean`: Group 6
- `v_loss_mean_step`: Group 6
- `wallclock_seconds_total`: Group 1
