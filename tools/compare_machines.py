import pandas as pd
import os 

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

csv_path = os.path.join(project_root, "data", "group10_2026.csv")
output_path = os.path.join(project_root, "data", "same_config_diff_machines.csv")

# upload csv file 
df = pd.read_csv(csv_path)

df.columns = df.columns.str.strip()

# create column from device characteristics
hw_cols = ["cpu_model", "cpu_cores", "ram_gb_total", "gpu_model"]
df["device_signature"] = (
    df["cpu_model"].astype(str).str.strip()
    + " | "
    + df["cpu_cores"].astype(str)
    + " cores | "
    + df["ram_gb_total"].astype(str)
    + "GB RAM | "
    + df["gpu_model"].astype(str).str.strip()
)

# columns which define the same configuration 
config_cols = [
    "environment_name",
    "batch_size",
    "buffer_size",
    "learning_rate",
    "hidden_units",
    "num_layers",
    "epsilon",
    "beta",
    "num_epoch",
    "max_steps",
    "reward_threshold",
    "seed",
]



# columns which define the  different results
result_cols = ["reached_threshold", "final_mean_reward", "steps_to_threshold"]

# group by configuration
diff_groups = []

for _, group in df.groupby(config_cols, dropna=False):
  if group["device_signature"].nunique() > 1:
    has_diff = any(group[rc].nunique(dropna=False) > 1 for rc in result_cols)
    if has_diff:
      diff_groups.append(group)

if diff_groups:
  result_df = pd.concat(diff_groups, ignore_index=True)

  result_df = result_df.sort_values(
      by=config_cols + ["device_signature", "seed"]
  )

  display_cols = (
      ["run_id", "environment_name", "seed"]
      + result_cols
      + hw_cols
      + [
          "batch_size",
          "buffer_size",
          "learning_rate",
          "hidden_units",
          "num_layers",
          "epsilon",
          "beta",
          "num_epoch",
      ]
  )

  result_df[display_cols].to_csv(output_path, index=False)

  print(f"Found {len(diff_groups)} configurations with different results on different devices!")
  print(f"Total selected rows: {len(result_df)}.")
  print(f"Result saved to: {output_path}\n")

  short_cols = [
      "run_id",
      "environment_name",
      "seed",
      "reached_threshold",
      "final_mean_reward",
      "steps_to_threshold",
      "cpu_model",
      "ram_gb_total",
  ]

  print("\n--- Summary by devices (on matching configurations) ---")
  summary = result_df.groupby("device_signature").agg(
      runs_count=("run_id", "count"),
      success_rate=(
          "reached_threshold",
          lambda x: f"{(x == True).mean() * 100:.1f}%",
      ),
      mean_final_reward=("final_mean_reward", "mean"),
      mean_steps=("steps_to_threshold", "mean"),
  )
  print(summary.to_string())
else:
  print("Not found configurations that were run on different devices and gave different results.")