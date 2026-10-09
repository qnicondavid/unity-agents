import csv
import platform
import subprocess
import time
import argparse
from pathlib import Path

import psutil
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
ML_AGENTS_DIR = REPO_ROOT / "external" / "ml-agents"
CSV_FILE = REPO_ROOT / "docs" / "runs" / "data.csv"

HEADERS = [
    "run_id", "environment_name", "algorithm",
    "cpu_model", "cpu_cores", "ram_gb_total", "gpu_model",
    "batch_size", "buffer_size", "learning_rate", "hidden_units",
    "num_layers", "max_steps", "wall_clock_time_sec",
    "peak_ram_mb", "final_mean_reward"
]


def check_headers():
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists() or CSV_FILE.stat().st_size == 0:
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
    else:
        with open(CSV_FILE, mode="r", newline="") as f:
            existing_headers = next(csv.reader(f), [])

        if existing_headers != HEADERS:
            raise ValueError(
                "Existing CSV headers do not match the expected columns."
            )


def get_hardware_info():
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
    config_path = Path(config_path)

    if config_path.is_absolute() and config_path.exists():
        return config_path

    # Relative paths are looked up inside the ml-agents submodule first
    submodule_path = ML_AGENTS_DIR / config_path

    if submodule_path.exists():
        return submodule_path

    if config_path.exists():
        return config_path.resolve()

    raise FileNotFoundError(
        f"Configuration file not found: {config_path} "
        f"(looked in {ML_AGENTS_DIR}). Is the submodule initialised?"
    )


def parse_yaml_config(config_path):
    full_config_path = resolve_config_path(config_path)

    with open(full_config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    behaviors = config.get("behaviors", {})

    if not behaviors:
        raise ValueError(
            f"No behaviors found in configuration file: {full_config_path}"
        )

    # Log only the first behavior defined in the config
    env_name = next(iter(behaviors))
    behavior_config = behaviors[env_name]

    trainer = behavior_config.get("trainer_type", "ppo").upper()
    hyperparameters = behavior_config.get("hyperparameters", {})
    network = behavior_config.get("network_settings", {})

    extracted_params = {
        "env_name": env_name,
        "algorithm": trainer,
        "batch_size": hyperparameters.get("batch_size", "N/A"),
        "buffer_size": hyperparameters.get("buffer_size", "N/A"),
        "learning_rate": hyperparameters.get("learning_rate", "N/A"),
        "hidden_units": network.get("hidden_units", "N/A"),
        "num_layers": network.get("num_layers", "N/A"),
        "max_steps": behavior_config.get("max_steps", "N/A")
    }

    return extracted_params


def get_final_mean_reward(run_id, behavior_name):
    # mlagents-learn runs from ML_AGENTS_DIR, so results is created there.
    event_dir = ML_AGENTS_DIR / "results" / run_id / behavior_name

    if not event_dir.exists():
        print(f"[WARNING] Results directory not found: {event_dir}")
        return None

    event_files = sorted(event_dir.glob("events.out.tfevents.*"))

    if not event_files:
        print(f"[WARNING] No TensorBoard event file found in {event_dir}")
        return None

    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator
    )

    try:
        accumulator = EventAccumulator(str(event_dir))
        accumulator.Reload()

        reward_tag = "Environment/Cumulative Reward"

        if reward_tag not in accumulator.Tags().get("scalars", []):
            print(f"[WARNING] Reward metric '{reward_tag}' not found")
            return None

        values = accumulator.Scalars(reward_tag)

        if not values:
            print("[WARNING] No reward values recorded")
            return None

        return float(values[-1].value)

    except Exception as e:
        print(f"[WARNING] Failed to extract reward: {e}")
        return None


def run_and_log_experiment(config_path, run_id):
    check_headers()

    # Read the config first so its settings can be included in the CSV row.
    full_config_path = resolve_config_path(config_path)
    parsed_params = parse_yaml_config(full_config_path)
    env_name = parsed_params["env_name"]

    cpu_model, cpu_cores, ram_gb, gpu_model = get_hardware_info()

    print(f"\n[INFO] Starting run: '{run_id}' for environment '{env_name}'...")

    start_time = time.time()
    peak_ram_mb = 0.0

    cmd = [
        "mlagents-learn",
        str(full_config_path),
        f"--run-id={run_id}"
    ]

    process = subprocess.Popen(cmd, cwd=ML_AGENTS_DIR)
    ps_proc = psutil.Process(process.pid)

    try:
        while process.poll() is None:
            time.sleep(2)

            try:
                # Include child processes because the trainer may launch workers.
                processes = [ps_proc] + ps_proc.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                processes = [ps_proc]

            mem_bytes = 0

            for tracked_process in processes:
                try:
                    mem_bytes += tracked_process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            current_ram_mb = mem_bytes / (1024 * 1024)
            peak_ram_mb = max(peak_ram_mb, current_ram_mb)

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted. Stopping the process...")
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    wall_time_sec = round(time.time() - start_time, 2)
    peak_ram_mb = round(peak_ram_mb, 2)

    final_mean_reward = get_final_mean_reward(run_id, env_name)

    row_data = {
        "run_id": run_id,
        "environment_name": env_name,
        "algorithm": parsed_params["algorithm"],
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "ram_gb_total": ram_gb,
        "gpu_model": gpu_model,
        "batch_size": parsed_params["batch_size"],
        "buffer_size": parsed_params["buffer_size"],
        "learning_rate": parsed_params["learning_rate"],
        "hidden_units": parsed_params["hidden_units"],
        "num_layers": parsed_params["num_layers"],
        "max_steps": parsed_params["max_steps"],
        "wall_clock_time_sec": wall_time_sec,
        "peak_ram_mb": peak_ram_mb,
        "final_mean_reward": (
            round(final_mean_reward, 4)
            if final_mean_reward is not None else ""
        )
    }

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row_data)

    reward_display = (
        round(final_mean_reward, 4)
        if final_mean_reward is not None else "Unavailable"
    )

    print(f"[SUCCESS] Appended results to {CSV_FILE}")
    print(
        f" -> Wall-Clock Time: {wall_time_sec}s"
        f" | Peak RAM: {peak_ram_mb} MB"
        f" | Mean Reward: {reward_display}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ML-Agents and record hardware/training metrics."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/ppo/3DBall.yaml",
        help="Config YAML, relative to external/ml-agents (or absolute)"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default="3DBall_test_dynamic",
        help="Run ID for training"
    )

    args = parser.parse_args()

    run_and_log_experiment(args.config, args.run_id)
