import argparse
import csv
import platform
import subprocess
from pathlib import Path

import psutil
import yaml

# set up directories relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
ML_AGENTS_DIR = REPO_ROOT / "external" / "ml-agents"
RESULTS_DIR = REPO_ROOT / "results"
CSV_FILE = REPO_ROOT / "data" / "group10_2026.csv"
CONFIG_DIR = REPO_ROOT / "data" / "configs"

REWARD_TAG = "Environment/Cumulative Reward"

# columns for our dataset csv
HEADERS = [
    "run_id", "environment_name", "seed",
    "batch_size", "buffer_size", "learning_rate", "hidden_units", "num_layers",
    "epsilon", "beta", "num_epoch",
    "max_steps", "summary_freq", "reward_threshold",
    "steps_to_threshold", "reached_threshold", "final_mean_reward",
    "cpu_model", "cpu_cores", "ram_gb_total", "gpu_model",
]

# maps hyperparameters to their sections in the ml-agents yaml config
TUNEABLE_SECTION = {
    "batch_size": "hyperparameters",
    "buffer_size": "hyperparameters",
    "learning_rate": "hyperparameters",
    "epsilon": "hyperparameters",
    "beta": "hyperparameters",
    "num_epoch": "hyperparameters",
    "hidden_units": "network_settings",
    "num_layers": "network_settings",
    "max_steps": None,
}


def check_headers():
    # make sure the csv file exists and has the correct header row
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with open(CSV_FILE, mode="w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        return

    with open(CSV_FILE, mode="r", newline="") as f:
        existing = next(csv.reader(f), [])

    if existing != HEADERS:
        raise ValueError(
            "Existing CSV headers do not match the expected columns. "
            "Move the old file aside and rerun."
        )


def get_hardware_info():
    # Get hardware details 
    cpu_model = platform.processor() or "Unknown CPU"
    cpu_cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    gpu_model = "N/A"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return cpu_model, cpu_cores, ram_gb, gpu_model


def resolve_config_path(config_path):
    # Determine yaml config file loaction
    config_path = Path(config_path)

    if config_path.is_absolute() and config_path.exists():
        return config_path

    submodule_path = ML_AGENTS_DIR / config_path
    if submodule_path.exists():
        return submodule_path

    if config_path.exists():
        return config_path.resolve()

    raise FileNotFoundError(
        f"Configuration file not found: {config_path} "
        f"(looked in {ML_AGENTS_DIR})."
    )


def build_config(base_config_path, run_id, tuneables):
    # Load the base yaml config, update with our specified hyperparams and add as a new config
    base_path = resolve_config_path(base_config_path)

    with open(base_path, "r") as f:
        config = yaml.safe_load(f) or {}

    behaviors = config.get("behaviors", {})
    if not behaviors:
        raise ValueError(f"No behaviors found in: {base_path}")

    behavior_name = next(iter(behaviors))
    behavior = behaviors[behavior_name]

    trainer = str(behavior.get("trainer_type", "ppo")).lower()
    if trainer != "ppo":
        raise ValueError(f"Expected PPO but trainer_type is '{trainer}'.")

    for key, value in tuneables.items():
        if value is None:
            continue
        section = TUNEABLE_SECTION[key]
        if section is None:
            behavior[key] = value
        else:
            behavior.setdefault(section, {})[key] = value

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG_DIR / f"{run_id}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return out_path, behavior_name, behavior


def get_reward_curve(run_id, behavior_name):
    # Get reward history from tensorboard (now in root results/ directory)
    event_dir = RESULTS_DIR / run_id / behavior_name

    if not event_dir.exists():
        print(f"[WARNING] Results directory not found: {event_dir}")
        return []

    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator
    )

    try:
        accumulator = EventAccumulator(str(event_dir))
        accumulator.Reload()

        if REWARD_TAG not in accumulator.Tags().get("scalars", []):
            print(f"[WARNING] Reward metric '{REWARD_TAG}' not found")
            return []

        return [(e.step, float(e.value)) for e in accumulator.Scalars(REWARD_TAG)]

    except Exception as e:
        print(f"[WARNING] Failed to read reward curve: {e}")
        return []


def steps_to_threshold(curve, threshold, window_size=3):
    # Sliding window for checking steps to a threshold to prevent false positives due performance spikes
    if len(curve) < window_size:
        for step, value in curve:
            if value >= threshold:
                return step
        return None

    for i in range(len(curve) - window_size + 1):
        window_values = [curve[i + j][1] for j in range(window_size)]
        if sum(window_values) / window_size >= threshold:
            return curve[i + window_size - 1][0]
            
    return None


def run_and_log_experiment(config_path, run_id, seed=0, tuneables=None,
                           threshold=80.0, env_path=None):
    check_headers()
    tuneables = tuneables or {}

    # prevent accidentally overwriting previous run data in root results/
    if (RESULTS_DIR / run_id).exists():
        raise FileExistsError(f"Results for run-id '{run_id}' already exist.")

    full_config_path, env_name, behavior = build_config(
        config_path, run_id, tuneables
    )
    hyper = behavior.get("hyperparameters", {})
    network = behavior.get("network_settings", {})

    cpu_model, cpu_cores, ram_gb, gpu_model = get_hardware_info()

    print(f"\n[INFO] Starting run '{run_id}' ({env_name}, seed {seed})")

    # Point to root results
    abs_results_dir = RESULTS_DIR.resolve()
    cmd = [
        "mlagents-learn", str(full_config_path),
        f"--run-id={run_id}", f"--seed={seed}",
        f"--results-dir={abs_results_dir}",
    ]
    if env_path:
        abs_env_path = Path(env_path).resolve()
        cmd.append(f"--env={abs_env_path}")

    process = subprocess.Popen(cmd, cwd=ML_AGENTS_DIR)

    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[WARNING] Interrupted. This run is NOT logged.")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return

    if process.returncode != 0:
        print(f"[ERROR] mlagents-learn exited with code {process.returncode}. "
              "This run is NOT logged.")
        return

    # Get results once training finishes
    curve = get_reward_curve(run_id, env_name)
    if not curve:
        print("[ERROR] No reward data found. This run is NOT logged.")
        return

    step = steps_to_threshold(curve, threshold)
    reached = step is not None

    # Group everything up into a dictionary for the csv row
    row = {
        "run_id": run_id,
        "environment_name": env_name,
        "seed": seed,
        "batch_size": hyper.get("batch_size", "N/A"),
        "buffer_size": hyper.get("buffer_size", "N/A"),
        "learning_rate": hyper.get("learning_rate", "N/A"),
        "hidden_units": network.get("hidden_units", "N/A"),
        "num_layers": network.get("num_layers", "N/A"),
        "epsilon": hyper.get("epsilon", "N/A"),
        "beta": hyper.get("beta", "N/A"),
        "num_epoch": hyper.get("num_epoch", "N/A"),
        "max_steps": behavior.get("max_steps", "N/A"),
        "summary_freq": behavior.get("summary_freq", "N/A"),
        "reward_threshold": threshold,
        "steps_to_threshold": step if reached else "",
        "reached_threshold": reached,
        "final_mean_reward": round(curve[-1][1], 4),
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "ram_gb_total": ram_gb,
        "gpu_model": gpu_model,
    }

    # append to our final csv file
    with open(CSV_FILE, mode="a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

    print(f"[SUCCESS] Appended results to {CSV_FILE}")
    print(f" -> Reached threshold: {reached}"
          f" | Steps to threshold: {step if reached else 'censored'}"
          f" | Final mean reward: {row['final_mean_reward']}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run ML-Agents PPO and log hyperparameters and steps to threshold."
    )
    p.add_argument("--config", default="config/ppo/3DBall.yaml")
    p.add_argument("--run-id", default="3DBall_test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=80.0)
    p.add_argument("--env-path", default=None)

    p.add_argument("--batch-size", type=int)
    p.add_argument("--buffer-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--hidden-units", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--epsilon", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--num-epoch", type=int)
    p.add_argument("--max-steps", type=int)

    a = p.parse_args()

    run_and_log_experiment(
        a.config, a.run_id, seed=a.seed,
        tuneables={
            "batch_size": a.batch_size,
            "buffer_size": a.buffer_size,
            "learning_rate": a.learning_rate,
            "hidden_units": a.hidden_units,
            "num_layers": a.num_layers,
            "epsilon": a.epsilon,
            "beta": a.beta,
            "num_epoch": a.num_epoch,
            "max_steps": a.max_steps,
        },
        threshold=a.threshold, env_path=a.env_path,
    )